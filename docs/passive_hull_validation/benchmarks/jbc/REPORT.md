# JBC Stage 1 result

## Completion artifact

```text
JBC GEOMETRY             BLOCKED (official raw IGES bytes unavailable)
JBC M1 PACKAGE           NOT RUN
JBC RUNTIME ROUNDTRIP    NOT RUN
JBC RESISTANCE           NOT SCORED (M1 package unavailable)
JBC SINKAGE              NOT SCORED (M1 package unavailable; restoring-only attitude is not tow equilibrium)
JBC TRIM                 NOT SCORED (M1 package unavailable; restoring-only attitude is not tow equilibrium)

GENERATOR MODIFIED       NO
```

The holdout did not progress to a MANTA prediction. Direct download from the official NMRI/Tokyo 2015 hosts failed in this runtime because DNS resolution is unavailable. The geometry pages identify the official submerged JBC geometry as IGES and list the later above-water IGES geometry. NMRI also lists its resistance/self-propulsion result spreadsheet but explicitly directs users to contact NMRI for the files. No unofficial or paper-derived hull mesh was substituted.

All 10 M1 implementation source hashes in `../../frozen_baseline_manifest.json` revision 2 were rechecked against the current source tree and match. The JBC frozen generator code hash therefore remains the manifest's recorded hash. No JBC package exists to hash; package hash-before-scoring is not applicable because generation never ran.

## Published reference values (source transcription only)

The published design-full model-scale particulars are Lpp 7.000 m, waterline beam 1.125 m, draft 0.4125 m, displacement 2.7870 m³, and bare-hull wetted area 12.2206 m². Official Case 1.1a is calm-water towing, bare hull without propeller or rudder, free heave and pitch, at U=1.179 m/s and Fr=0.142. The workshop defines

`CT = RT / (0.5 rho U^2 S0)` with `S0/Lpp^2=0.2494`.

The accessible published EFD summary gives CT=4.29e-3, sinkage/Lpp=-0.0857% (positive upward), and trim/Lpp=-0.180% (positive bow-up). At model scale those motions correspond to -0.005999 m and -0.01260 m, respectively. No explicit experimental uncertainty interval was found in the accessible published summaries. The NMRI test-summary PDF cross-check reports CT=4.289e-3, sinkage/Lpp=-0.086%, and trim/Lpp=-0.180%; the displayed difference in CT reflects source precision/rounding, not a new score.

These are benchmark comparator values, not MANTA predictions. Since no official source geometry was acquired and no unchanged M1 package was generated, absolute/relative MANTA errors cannot be reported. The machine-readable transcriptions and their source URLs are in `reference/jbc_reference.json`; that file explicitly records that original source bytes and hashes are absent.

## Provenance and next input

- Tokyo 2015 JBC geometry and conditions page: https://www.t2015.nmri.go.jp/jbc_gc.html
- Case 1.1a condition and normalization page: https://t2015.nmri.go.jp/Instructions_JBC/Case_1-1a.html
- NMRI above-water JBC geometry page and IGES listing: https://www.nmri.go.jp/en/study/intellectual/db/jbc/
- NMRI resistance/self-propulsion data listing: https://www.nmri.go.jp/study/intellectual/db/jbc2/
- NMRI Case 1.1 towing test summary: https://t2015.nmri.go.jp/Presentations/Day1-AM4-JBC-Resist_etc-Larsson.pdf
- Secondary published transcription used only as cross-check: https://dcwan.sjtu.edu.cn/userfiles/Tokyo_JBC_SUNTAO_YINCH_WUJW.pdf

To finish this holdout, stage the official `JBC_IGES.zip` from Tokyo 2015, the official above-water IGES if required for closure, and the NMRI resistance-results workbook. Then reconstruct/qualify a single watertight full hull, generate through frozen M1, archive and hash the package, run the runtime roundtrip, and only then calculate the Case 1.1a errors. The current frozen model's hydrostatic restoring alone does not predict dynamic free heave/pitch in a towing flow, so sinkage and trim should remain not scored unless an appropriate supported equilibrium prediction is present.
