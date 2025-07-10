# TUM MLMI - DMCAF

# Time Planing
```mermaid
gantt
    title Time-plan
    dateFormat  YYYY-MM-DD
    tickInterval 1week
    weekday monday

    section Concept Phase
        Lit. Rev. Diffusion Models (DMs) : done, c1, 2025-05-12, 3w
        Run DMs and Vis. Tools : done, c2, 2025-05-19, 17d
        Lit. Rev. Eval Frameworks: done, c3, 2025-05-26, 3w
        Prep. Slides: done, R1, 2025-06-05, 11d

    section Implementation
        MVP Implementation: i1, after m1, 2w
        Specific Metrics: i2, 2025-06-23, 2w
        Vis. Tools and Interact. GUI: i4, 2025-06-23, 3w
        Advanced Conditioning: i3,  2025-06-30, 2w
        Framework Refinement: i5, after i3, 4w

    section Experimentation
        Design & Run Exp. I: ex1, 2025-06-30 ,2w
        Design & Run Exp. II: ex2, 2025-07-07 ,3w
        Comprehensive Evaluation: wp6, 2025-07-14, 6w
        Run Final Experiments: ex3, after m2 ,2w

    section Reporting
        Prep. Final Slides: r1, 2025-07-21, 2w
        Finalize Report: r2, 2025-08-18, 2w

    section Actual
        Hackathon: h1, 2025-07-18, 3d
        Final Presentation rehearsal: milestone, m3, 2025-07-31, 0d

    section Milestones
        Mid-Presentation: milestone, m1, 2025-06-16, 0d
        Final Presentation: milestone, m2, 2025-08-04, 0d
        Final Submission: milestone, m3, 2025-09-08, 0d
```

# Tasks before Hackathon
- [ ] Familiarize with diffusors lib. + Define interfaces for Framework (Mock functions)
- [ ] Make DM runner work (start with one config)
- [ ] Metrics research e.g. FID Which pertained models exists which datasets are used for
- [ ] What exists specifically about medical images