//! Offline bridge only: one persistent voice per process, one JSON response per request.
//! No changes to the engine or its real-time rendering contract.
use chimera::anatomy::human::{ArticKey, ArticPatch, NEUTRAL};
use chimera::voice::human_voice::{HumanVoice, SourcePatch};
use serde::Deserialize;
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};

const FS: usize = 48000;
const FRAMES: usize = 2400; // 50 ms actions; engine dynamics still run at control rate.
const KEYS: [ArticKey; 8] = [
    ArticKey::TbX,
    ArticKey::TbY,
    ArticKey::Jaw,
    ArticKey::LipAp,
    ArticKey::LipProt,
    ArticKey::LipWidth,
    ArticKey::Root,
    ArticKey::Larynx,
];
// Vowel exploration subset of calibrate.rs bounds. No independent tongue-tip control yet.
const LO: [f64; 8] = [1.2, 5.0, 0.05, 0.12, 0.0, 0.0, -0.9, -1.3];
const HI: [f64; 8] = [4.6, 8.8, 2.3, 2.0, 1.4, 1.0, 0.9, 1.3];

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    Schema,
    Reset { f0: f64, targets: Option<[f64; 8]> },
    Step { targets: [f64; 8] },
}

fn patch(targets: [f64; 8]) -> Result<ArticPatch, String> {
    let mut p = ArticPatch::default();
    for i in 0..8 {
        if !targets[i].is_finite() || !(-1.0..=1.0).contains(&targets[i]) {
            return Err("targets must be finite normalized values in [-1, 1]".into());
        }
        p.set(KEYS[i], LO[i] + (targets[i] + 1.0) * 0.5 * (HI[i] - LO[i]));
    }
    Ok(p)
}

fn normalized(voice: &HumanVoice) -> Vec<f64> {
    KEYS.iter()
        .enumerate()
        .map(|(i, k)| 2.0 * (voice.artic.x.get(*k) - LO[i]) / (HI[i] - LO[i]) - 1.0)
        .collect()
}

fn handle(req: Request, voice: &mut Option<HumanVoice>) -> Result<Value, String> {
    match req {
        Request::Schema => Ok(json!({"version": 1, "sample_rate": FS, "frames": FRAMES,
            "keys": KEYS.map(|k| k.name()), "low": LO, "high": HI,
            "neutral": KEYS.iter().enumerate().map(|(i,k)|
                (2.0 * (NEUTRAL.get(*k)-LO[i])/(HI[i]-LO[i])-1.0).clamp(-1.0,1.0)).collect::<Vec<_>>() })),
        Request::Reset { f0, targets } => {
            if !f0.is_finite() || !(70.0..=400.0).contains(&f0) {
                return Err("f0 must be between 70 and 400 Hz".into());
            }
            let p = targets.map(patch).transpose()?;
            let mut v = HumanVoice::with_defaults(FS as f64);
            v.set_source(&SourcePatch {
                f0: Some(f0),
                glide: Some(0.0),
                roughness: Some(0.0),
                drift: Some(0.0),
                ..Default::default()
            });
            if let Some(p) = p {
                v.artic.snap(&p);
            }
            v.set_gate(1.0);
            // Let gate, source, acoustic state and tongue tip settle before scoring.
            v.render(&mut vec![0.0; FS / 4]);
            *voice = Some(v);
            render(voice.as_mut().unwrap())
        }
        Request::Step { targets } => {
            let p = patch(targets)?;
            let v = voice.as_mut().ok_or("reset before step")?;
            v.set_articulation(&p, Some(true));
            render(v)
        }
    }
}

fn render(v: &mut HumanVoice) -> Result<Value, String> {
    let mut audio = vec![0.0f32; FRAMES];
    v.render(&mut audio);
    let positions = normalized(v);
    if audio.iter().any(|x| !x.is_finite()) || positions.iter().any(|x| !x.is_finite()) {
        return Err("non-finite engine output".into());
    }
    Ok(json!({"audio": audio, "positions": positions}))
}

fn main() -> io::Result<()> {
    let mut voice = None;
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let result = serde_json::from_str::<Request>(&line?)
            .map_err(|e| e.to_string())
            .and_then(|r| handle(r, &mut voice));
        let response = result.unwrap_or_else(|error| json!({"error": error}));
        serde_json::to_writer(&mut stdout, &response)?;
        writeln!(stdout)?;
        stdout.flush()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_invalid_controls_and_step_before_reset() {
        assert!(patch([2.0; 8]).is_err());
        assert!(patch([f64::NAN; 8]).is_err());
        assert!(handle(Request::Step { targets: [0.0; 8] }, &mut None).is_err());
        assert!(
            handle(
                Request::Reset {
                    f0: 0.0,
                    targets: None
                },
                &mut None
            )
            .is_err()
        );
    }
    #[test]
    fn reset_reproduces_audio_after_movement() {
        let mut v = None;
        let a = handle(
            Request::Reset {
                f0: 140.0,
                targets: None,
            },
            &mut v,
        )
        .unwrap();
        handle(Request::Step { targets: [0.0; 8] }, &mut v).unwrap();
        let b = handle(
            Request::Reset {
                f0: 140.0,
                targets: None,
            },
            &mut v,
        )
        .unwrap();
        assert_eq!(a, b);
        assert_eq!(a["audio"].as_array().unwrap().len(), FRAMES);
        assert!(
            a["audio"]
                .as_array()
                .unwrap()
                .iter()
                .any(|x| x.as_f64().unwrap().abs() > 1e-6)
        );
    }
}
