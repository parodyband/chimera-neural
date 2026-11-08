# Neural articulatory learning: research and first decision

Research date: 2026-11-08. Base: CHIMERA `5ab44de`. This is a new experimental fork;
no claim of a trained controller or demonstrated speech improvement is made.

## Decision

Begin with **goal-conditioned Soft Actor-Critic (SAC)**: a stochastic continuous-action
actor and two Q critics, each using a small two-layer MLP (128 units per hidden layer).
Inputs describe the desired acoustics and recent articulator/acoustic state. Outputs move
the articulators; the real Rust simulator supplies sound and reward. Replay carries what
was learned from one attempt into the next. No gradients through Rust are needed.

This is our engineering starting choice, not a published result that SAC is best for CHIMERA.
The neural architecture and the learning algorithm are separate choices: an MLP is enough
for the first compact observations; a temporal encoder or recurrent model becomes more
useful as targets become syllables and phrases. A large transformer is not needed to test
whether exploration learns reusable movement-to-sound relationships.

The longer-term candidate closest to “learn the mouth, imagine attempts, then act” is
**goal-conditioned world-model control**, with TD-MPC2 as a concrete reference. First
establish a reliable environment and SAC baseline so model error and environment/reward
errors can be distinguished. The current implementation contains SAC only.

## Primary evidence and applicability

1. [Shitov et al., 2023: Deep Reinforcement Learning for Articulatory Synthesis in a
   Vowel-to-Vowel Imitation Task](https://pmc.ncbi.nlm.nih.gov/articles/PMC10147087/).
   Directly relevant physical-synthesizer precedent: learns control with a neural dynamics
   model and acoustic embeddings. Supports trying model-based learning and highlights
   how much the acoustic representation matters. It is a restricted imitation experiment,
   not evidence of arbitrary natural speech or a guarantee for this simulator.
2. [Anand et al., 2025: Teaching Machines to Speak using Articulatory
   Control](https://arxiv.org/html/2510.05619v1).
   Uses PPO for articulatory movement and syllable-level feedback. Important distinction:
   trajectories become sound through pretrained SPARC, a learned decoder, rather than
   CHIMERA-like physical acoustics. Its six-syllable results support the control framing,
   but cannot establish the expressivity or learning cost of our Rust engine.
3. [Haarnoja et al., 2018: Soft Actor-Critic](https://arxiv.org/abs/1801.01290).
   Off-policy maximum-entropy actor–critic for continuous actions. Replay and stochastic
   exploration suit repeated simulated attempts. [SB3's maintained SAC implementation](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
   supplies the baseline instead of writing an RL algorithm from scratch. Its standard
   policies are not recurrent, so this scaffold uses frame history.
4. [Hansen et al., 2024: TD-MPC2](https://arxiv.org/abs/2310.16828).
   Learns a latent world model and plans trajectories through it. This is the strongest
   follow-up candidate when simulator interaction cost or anticipation limits SAC.
   Its broad continuous-control benchmarks are not speech benchmarks. We still need a
   target-audio encoder, appropriate state history and validation against real renders.
5. [Hafner et al.: DreamerV3](https://arxiv.org/abs/2301.04104).
   Learns behavior through imagined world-model trajectories. Worth considering for longer
   sequences and partial observability, but its recurrent latent-model machinery adds more
   moving parts than the initial vowel task needs. It is not ruled out by evidence here.
6. [Andrychowicz et al., 2017: Hindsight Experience Replay](https://arxiv.org/abs/1707.01495).
   A failed attempt can train reaching the goal it actually achieved. Attractive for
   babbling: “I missed /a/, but learned how to reach this other sound.” Original evidence
   concerns sparse goal-reaching rewards. Our reward is dense; measure the benefit before
   adopting it. Relabel rewards and every goal-conditioned observation consistently;
   moving acoustic targets require more care than stationary goals. Not implemented yet.
7. [Engel et al., 2020: DDSP](https://arxiv.org/abs/2001.04643).
   Shows the value of structured synthesis in neural learning. It does not make the
   existing Rust airway differentiable. A differentiable frequency-domain airway model
   could assist sonorant fitting; it cannot stand in for all transient aerodynamics.
8. [Chen, Lee and Sitzmann, 2026: Reconstructing the Vocal Tract with Differentiable
   Acoustic Simulation](https://echen01.github.io/vocal-tract/).
   Recent alternative: differentiable frequency-domain tube acoustics and neural fields
   fit sound to moving tract geometry. This makes differentiable physical modeling a
   serious parallel research direction, beyond generic DDSP. It is a different acoustic
   model, not automatic differentiation of CHIMERA's moving-boundary airway. Its fitting
   demos do not by themselves demonstrate a trial-and-error policy that transfers across
   targets. Revisit it if a differentiable surrogate becomes the limiting requirement.

## Alternatives and what would change the choice

PPO is a credible simpler on-policy comparison, with directly relevant syllable work above.
SAC's reuse of simulator experience is the reason to start there, not a claim that PPO
cannot work. AlphaZero-style discrete tree search would require an awkward discretization
of continuous mouth movements; use continuous-action control and optional learned planning
to preserve the user's core idea. Supervised audio-to-controls is useful after successful
attempts provide good training data; it is not required before trial-and-error learning.
Direct numerical fitting remains a useful evaluation baseline, not the core learning system.

Choose TD-MPC2 or a small ensemble dynamics model if SAC improves but needs too many real
renders. Add recurrence/temporal encoding when frame history fails on transitions. Add HER
if many failed attempts contain useful achieved goals. Do not expand to a large model merely
because a reward is exploitable or the engine cannot reproduce the target.

## First experiment and evidence threshold

Start with steady synthetic vowel goals, fixed source and anatomy, and eight controls.
Separate train seeds from evaluation seeds. Run several training seeds and compare learned,
neutral, random, and eventually render-budget-matched numerical-search controls. The first
question is whether one policy reaches fresh goals better after experience. This tests
transfer within the goal distribution, not human speech imitation or anatomical recovery.

The scaffold provides the first three evaluation policies. A short setup smoke run does
not answer that learning question. Proposed first budget: 20,000 steps per training seed,
three seeds, measured locally before estimating a larger or paid run. Stop and inspect if
it cannot outperform neutral on synthetic goals; inspect reward, exploration and state
before blaming model size or physics.

Next use one speaker's repeated real vowels with supplied pitch, and separate recordings
for validation and final testing. Compare distance to the speaker's own repeated tokens,
listen, and inspect movements. Then add pitch trajectories, speaker anatomy, vowel glides,
nasals and syllables. Continuous target sequences need temporal losses/alignment rather
than averaging a whole recording into one spectrum. A sequence encoder (small causal CNN
or GRU) and a recurrent world model are plausible extensions, not tested decisions.

## Loss and scientific traps

The initial normalized log-mel shape plus RMS reward provides dense feedback. It is not
full multi-resolution STFT loss, phonetic accuracy, or human preference. Later combine
spectral envelopes and multi-resolution timing information with voiced/unvoiced and pitch
features. A speech embedding can reduce speaker nuisance but may ignore timbre or invite
adversarial artifacts; retain acoustic checks and listening. Do not train on Whisper WER.

Source pitch and microphone coloration can dominate spectral error. Supply measured pitch
first; fit speaker anatomy separately from utterance actions. Any future recording-chain
correction must be constrained and held out, or it can hide genuine engine errors. Audio
does not identify a unique tongue shape. Bound actions, favor smoothness, inspect anatomy.

The engine has state: pressure, source phase, moving boundaries. Never reset it between
actions, and never label a snapshot as the complete Markov state. Its documented geometry
and source limitations can be exploited by a policy. Failure to fit is not proof of missing
physics until optimization, pitch, alignment and reward design have been checked.

Shared physical parameters can later be fitted across speakers and utterances, but should
not silently become per-utterance cheats. Engine changes still follow the hill-climb
physiology rules, existing judges, goldens and speed guards. This fork adds an offline
environment and neural policy without changing the physical engine, generated tables,
judges, browser ABI, or UI.
