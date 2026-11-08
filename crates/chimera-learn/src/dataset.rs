//! Offline supervised data: the planner speaks random English word strings, and every control
//! frame it gave the voice is saved beside the audio. `replay` renders such a control track
//! back through the voice, the path a learned audio -> controls model is heard through.
//!
//! A control frame covers `HOP` output samples (5 control ticks, 6.67 ms at 48 kHz). It holds
//! the mean blended value, weight and speed of every control key over those ticks, then a flag
//! (1 when a score drove the voice). Files are little-endian f32, `FRAME` floats per frame.
use chimera::anatomy::human::BODY_MALE;
use chimera::control::score::{Blended, N_CONTROL};
use chimera::phonetics::Intonation;
use chimera::phonetics::english::EnglishG2P;
use chimera::phonetics::planner::{PhraseInput, PlanOptions, plan_phrases};
use chimera::voice::human_voice::{HumanVoice, SourcePatch};
use serde_json::json;
use std::io::{Read, Write};
use std::path::Path;

pub const FS: usize = 48000;
/// One control tick is 64 output samples at 2x oversampling; a frame is five.
pub const TICK: usize = 64;
pub const TICKS: usize = 5;
pub const HOP: usize = TICK * TICKS;
pub const FRAME: usize = 3 * N_CONTROL + 1;

/// xorshift64*: deterministic per (seed, index), no dependency.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn dictionary(repo: &Path) -> Result<(EnglishG2P, Vec<String>), String> {
    let path = repo.join("public/data/cmudict.txt.gz");
    let gz = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut text = String::new();
    flate2::read::GzDecoder::new(&gz[..]).read_to_string(&mut text).map_err(|e| e.to_string())?;
    // Plain alphabetic headwords; alternates like "read(2)" and abbreviations are skipped.
    let words = text
        .lines()
        .filter_map(|l| l.split('\t').next())
        .filter(|w| (2..=10).contains(&w.len()) && w.chars().all(|c| c.is_ascii_lowercase()))
        .map(str::to_string)
        .collect();
    let mut g = EnglishG2P::new();
    g.load_dict(&text);
    Ok((g, words))
}

pub fn write_wav(path: &Path, audio: &[f32]) -> std::io::Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    let n = audio.len() as u32 * 2;
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + n).to_le_bytes())?;
    f.write_all(b"WAVEfmt ")?;
    for x in [16u32] {
        f.write_all(&x.to_le_bytes())?;
    }
    for x in [1u16, 1] {
        f.write_all(&x.to_le_bytes())?;
    }
    f.write_all(&(FS as u32).to_le_bytes())?;
    f.write_all(&(FS as u32 * 2).to_le_bytes())?;
    for x in [2u16, 16] {
        f.write_all(&x.to_le_bytes())?;
    }
    f.write_all(b"data")?;
    f.write_all(&n.to_le_bytes())?;
    for x in audio {
        let s = (x.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        f.write_all(&s.to_le_bytes())?;
    }
    Ok(())
}

/// Render the voice frame by frame, recording each frame's mean control blend.
fn render_recorded(v: &mut HumanVoice, frames: usize) -> (Vec<f32>, Vec<f32>) {
    let mut audio = vec![0f32; frames * HOP];
    let mut ctl = Vec::with_capacity(frames * FRAME);
    for f in 0..frames {
        let mut acc = [0f64; FRAME];
        for t in 0..TICKS {
            let at = (f * TICKS + t) * TICK;
            v.render(&mut audio[at..at + TICK]);
            let (b, scored) = v.blend();
            if scored {
                for k in 0..N_CONTROL {
                    acc[k] += b[k].value;
                    acc[N_CONTROL + k] += b[k].weight;
                    acc[2 * N_CONTROL + k] += b[k].speed;
                }
                acc[FRAME - 1] += 1.0;
            }
        }
        let n = acc[FRAME - 1];
        for k in 0..FRAME - 1 {
            ctl.push(if n > 0.0 { (acc[k] / n) as f32 } else if k >= 2 * N_CONTROL { 1.0 } else { 0.0 });
        }
        ctl.push((n / TICKS as f64) as f32);
    }
    (audio, ctl)
}

/// Drive the voice from a control track. A frame whose flag is under 0.5 lets it rest.
pub fn render_replay(v: &mut HumanVoice, ctl: &[f32]) -> Vec<f32> {
    let frames = ctl.len() / FRAME;
    let mut audio = vec![0f32; frames * HOP];
    for (f, c) in ctl.chunks_exact(FRAME).enumerate() {
        if c[FRAME - 1] < 0.5 {
            v.set_blend_override(None);
        } else {
            let mut b = [Blended::default(); N_CONTROL];
            for (k, x) in b.iter_mut().enumerate() {
                *x = Blended {
                    value: c[k] as f64,
                    weight: c[N_CONTROL + k].max(0.0) as f64,
                    speed: c[2 * N_CONTROL + k].max(0.0) as f64,
                };
            }
            v.set_blend_override(Some(b));
        }
        v.render(&mut audio[f * HOP..(f + 1) * HOP]);
    }
    audio
}

fn bytes(x: &[f32]) -> Vec<u8> {
    x.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// `dataset <repo> <outdir> <seed> <count>`: `count` utterances, ids seed*1e6 + i.
pub fn dataset(args: &[String]) -> Result<(), String> {
    let [repo, out, seed, count] = args else {
        return Err("usage: chimera-learn dataset <repo> <outdir> <seed> <count>".into());
    };
    let seed: u64 = seed.parse().map_err(|_| "seed")?;
    let count: usize = count.parse().map_err(|_| "count")?;
    let out = Path::new(out);
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let (g, words) = dictionary(Path::new(repo))?;
    let mut meta = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out.join("meta.jsonl"))
        .map_err(|e| e.to_string())?;
    for i in 0..count {
        let id = seed * 1_000_000 + i as u64;
        let mut r = Rng::new(id + 1);
        let n_words = 3 + r.below(8);
        let mut text = String::new();
        for w in 0..n_words {
            text.push_str(&words[r.below(words.len())]);
            text.push_str(if w + 1 == n_words { "" } else if r.unit() < 0.12 { ", " } else { " " });
        }
        text.push(if r.unit() < 0.2 { '?' } else { '.' });
        let f0 = r.range(85.0, 150.0);
        let rate = r.range(0.8, 1.25);
        let accent = r.range(0.05, 0.3);
        let breathiness = r.range(0.0, 0.35);
        let tension = r.range(0.35, 0.7);
        let phrases = g.to_phrases(&text);
        let inputs: Vec<PhraseInput> = phrases
            .iter()
            .map(|p| PhraseInput { segments: p.segments.clone(), intonation: Some(p.intonation), pause: Some(p.pause) })
            .collect();
        let plan = plan_phrases(
            &inputs,
            &PlanOptions { body: Default::default(), t0: 0.05, f0: Some(f0), rate: Some(rate), accent: Some(accent), ..Default::default() },
        );
        let mut v = HumanVoice::new(FS as f64, BODY_MALE, 2);
        v.set_source(&SourcePatch { f0: Some(f0), breathiness: Some(breathiness), tension: Some(tension), ..Default::default() });
        v.schedule(&plan.gestures);
        let frames = ((plan.end + 0.25) * FS as f64 / HOP as f64).ceil() as usize;
        let (audio, ctl) = render_recorded(&mut v, frames);
        if audio.iter().any(|x| !x.is_finite()) {
            return Err(format!("{id}: non-finite audio"));
        }
        write_wav(&out.join(format!("{id}.wav")), &audio).map_err(|e| e.to_string())?;
        std::fs::write(out.join(format!("{id}.ctl")), bytes(&ctl)).map_err(|e| e.to_string())?;
        let ipa: Vec<String> = phrases.iter().map(|p| p.segments.iter().map(|s| s.ipa.clone()).collect()).collect();
        let q = phrases.iter().any(|p| p.intonation == Intonation::Question);
        writeln!(
            meta,
            "{}",
            json!({"id": id, "text": text, "ipa": ipa, "question": q, "frames": frames, "f0": f0, "rate": rate,
                   "accent": accent, "breathiness": breathiness, "tension": tension})
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// `replay <in.ctl> <out.wav> [f0]`: render a control track with the default male voice.
pub fn replay(args: &[String]) -> Result<(), String> {
    let (input, output) = match args {
        [i, o] | [i, o, _] => (i, o),
        _ => return Err("usage: chimera-learn replay <in.ctl> <out.wav> [f0]".into()),
    };
    let raw = std::fs::read(input).map_err(|e| e.to_string())?;
    if raw.len() % (4 * FRAME) != 0 {
        return Err(format!("{input}: not a whole number of {FRAME}-float frames"));
    }
    let ctl: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    if ctl.iter().any(|x| !x.is_finite()) {
        return Err("non-finite control".into());
    }
    let mut v = HumanVoice::new(FS as f64, BODY_MALE, 2);
    if let Some(f0) = args.get(2) {
        v.set_source(&SourcePatch::f0(f0.parse().map_err(|_| "f0")?));
    }
    let audio = render_replay(&mut v, &ctl);
    write_wav(Path::new(output), &audio).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Per-tick blends are a complete description of what the planner told the voice.
    #[test]
    fn per_tick_override_is_bit_exact() {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let (g, _) = dictionary(&repo).unwrap();
        let phrases = g.to_phrases("She said a quick bat.");
        let inputs: Vec<PhraseInput> =
            phrases.iter().map(|p| PhraseInput { segments: p.segments.clone(), intonation: Some(p.intonation), pause: Some(p.pause) }).collect();
        let plan = plan_phrases(&inputs, &PlanOptions { t0: 0.05, f0: Some(112.0), ..Default::default() });
        let mut v = HumanVoice::new(FS as f64, BODY_MALE, 2);
        v.set_source(&SourcePatch::f0(112.0));
        v.schedule(&plan.gestures);
        let ticks = ((plan.end + 0.25) * FS as f64 / TICK as f64) as usize;
        let mut a = vec![0f32; ticks * TICK];
        let mut rec = Vec::new();
        for t in 0..ticks {
            v.render(&mut a[t * TICK..(t + 1) * TICK]);
            let (b, s) = v.blend();
            rec.push((*b, s));
        }
        let mut w = HumanVoice::new(FS as f64, BODY_MALE, 2);
        w.set_source(&SourcePatch::f0(112.0));
        let mut out = vec![0f32; ticks * TICK];
        // the blend read after tick t was computed AT tick t's start: set it before rendering t
        for t in 0..ticks {
            w.set_blend_override(if rec[t].1 { Some(rec[t].0) } else { None });
            w.render(&mut out[t * TICK..(t + 1) * TICK]);
        }
        let first = a.iter().zip(&out).position(|(x, y)| x != y);
        assert_eq!(first, None, "first differing sample");
    }
}
