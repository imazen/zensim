# D3 viewing conditions — part 2 of 2 (AIC-3, AIC-4, SDR25, UPIQ, search log)

> Deliverable 3 of lane E2a (zensim Rev4, 2026-09-23). Verbatim passages from each dataset's own paper, gathered by a read-only helper agent and spot-checked by the lane (CID22 0.0213°, KonJND 30 cm, UPIQ 51 ppd, Korshunov 3.2 H re-found in the sources). Split in two parts to respect the 30 KB rule; full original: `/var/tmp/rev4-e2a/d3_viewing_conditions.md`.

## AIC-3 dataset (Testolina et al., QoMEX 2023; 10 sources)

Source: "JPEG AIC-3 Dataset: Towards Defining the High Quality to Nearly Visually Lossless Quality Range". `P:6a/6a48358a61ecac3b6dcc689eab04292e892687c390bb1a960718cc76673b3469.pdf`.

- **Setting:** crowd, expert viewers. p3: "The experiments were conducted in a crowdsourcing environment with expert viewers. The selected platform for the experiment was "QualityCrowd 2" [20]."
- **Screen:** p3: "Prior to the beginning of the experiment, a preliminary screen-check was conducted, and only subjects with a screen of size 1920 × 1080 or larger, with retina mode disabled (device pixel ratio equal to 1), were able to proceed to the experiment."
- **Crop / native:** p3: "Image cropping to a size of 945 × 880 was necessary in order to fit two stimuli side-by-side on the target screen size. The cropping was performed by selecting the salient area of each image or the area where the artifacts are the most visible". Table I (p2) lists reference resolutions from 560×888 to 2592×1946 with crop coordinates. Protocol: side-by-side pair comparison (p3).
- **Display model, distance, luminance:** not stated.
- **ppd:** not derivable. Presentation is at DPR 1 (one image px = one device px), native, cropped.
- **Relation to zensim:** zensim `DATA_SPLITS.md:233` calls the eval set "AIC-3 CTC (600 pairs, 10 refs)" and says "Mohammadi 2025 evaluates metrics on it". But Mohammadi et al. 2025 (`P:e1/e13e1076…pdf` p5) describe their "JPEG AIC-3 dataset" as "five high-quality source … cropped to a resolution of 620 × 800 pixels prior to testing". That is the 5-source BTC/PTC study below, not this 10-source QoMEX'23 study. **Which human study produced the labels behind zensim's 10-ref "AIC-3 CTC" corpus should be pinned before assigning it a viewing geometry.** I did not open the label files, per the brief.
- **Confidence:** high for the quotes; the provenance mapping is unresolved.

## AIC-3 BTC/PTC study (Testolina et al., DCC 2025; 5 sources) — the source of the AIC-4 sample labels

Source: "Fine-Grained Subjective Visual Quality Assessment for High-Fidelity Compressed Images". `P:a4/a4f8f336e1f38a57f42a65765b45f57612259f951dda100b6a37b660b93a5d2f.pdf`. Repo README: `/mnt/v/repos/iqa-tools/jpeg-aic__dataset-BTC-PTC-24/README.md`.

- **Setting:** crowd, AMT. p5: "Test subjects were recruited through Amazon Mechanical Turk (MTurk). The BTC and PTC experiments were conducted separately, with several months in between". README L7: "The study was conducted on Amazon Mechanical Turk (MTurk) from January 4 to January 10, 2024."
- **Crop:** p4: "Five images from the JPEG AIC-3 dataset were selected … and cropped to a size of 620 × 800 pixels".
- **BTC boosting (magnified, not native):** p3: "• Zooming: Images are first cropped to half the size in both dimensions and then upscaled to the initial size using Lanczos resampling. … • Artefact amplification: The pixel-wise difference between the original and distorted stimuli are linearly scaled in the three color channels separately with an amplification factor of 2. • Flicker: The test images are temporarily interleaved with the reference image at a change rate of 10 Hz". Layout: "The two test stimuli Ii and Ik are displayed side-by-side, each alternating with the source".
- **PTC (plain):** p3–4: "the decoded images are left untouched and used in place of the boosted versions. Source images are shown in-place with the test images, but the flicker is replaced by a toggle button … the maximal toggle frequency is limited to 2 Hz."
- **Display, resolution requirement, DPR, distance, luminance:** not stated in the paper. The README logs per-worker data rather than a controlled condition: L46 "`resolution` … Screen resolution of the worker's device during the task." L60 "`display_size` … Physical size of the worker's display device in inches".
- **zensim measurement (not a paper claim):** `/home/lilith/work/zen/zensim/benchmarks/hfhuman_2026-09-01.md:57` "BTC stimuli are *boosted* — the workers judged a 2× magnified, ~1.8× distortion-amplified rendering (measured, §2.4)". `/mnt/v/output/zensim/hfhuman-2026-09-01/_MANIFEST.json`: "PTC crops as displayed (native scale, native amplitude; G1 pixel-exact crop of the CTC source)".
- **ppd:** not derivable. BTC ppd is half of whatever the worker's native ppd was, because of the 2× zoom.
- **Confidence:** high that the geometry is unstated.

## AIC-4 sample / AIC-4 CTC

Sources: `~/tmp/papers-dvifm/wg1n101246-108-ICQ-Common_Test_Conditions_on_Objective_Quality_Assessment_v2_0.md` (CTC v2.0), the AIC-4 CfP `wg1n101157-…md`, the AIC white paper `wg1n101547-112-COM-White_Paper_on_JPEG_AIC.md`, and `/mnt/v/repos/iqa-tools/jpeg-aic__JPEG-AIC-4-datasets/README.md`.

- **Stimuli and human study:** CTC L83: "The source images in the example dataset correspond to images 2, 6, 7, 9, 10 of the AIC[-3 dataset]." README L27–31 points to the Jenadeleh 2025 (SDR25) and Testolina DCC 2025 papers for the "Subjective study and the scale reconstruction methods". README L36: "`PTC_images`: This folder contains the cropped versions of the test images that were used for the subjective quality assessmnet." So the human viewing conditions are those of the AIC-3 BTC/PTC crowd study above, and the paper leaves them unstated. The PTC crops are native; BTC is 2× zoomed.
- **CTC viewing parameters are for the anchor metrics, not the human study.** CTC L186: "For SDR images, the metric is configured for an effective display resolution of 37.84 pixels per degree and a peak brightness of 203 cd/m². For HDR images, the metric is configured for an effective display resolution of 56.55 pixels per degree, a peak brightness of 1000 cd/m²". CTC L188 (ColorVideoVDP; the md text is garbled across a column break): "For SDR images, the metric i[s configured for a standard Full HD display resolution (]-d standard_fhd), resulting in an effective display resolution of 37.84 pixels per degree, a peak brightness of 200 cd/m², a black level of 0.2 cd/m², and a reflec[ted light level of] 0.3979 cd/m². For HDR images, the metric is configured for an HDR display with an effective display resolution of 56.55 pixels per degree, a peak brightness of 1000 cd/m², a black level of 0.001 cd/m², and a reflected light level of 0.007958 cd/m²." The bracketed fragments are recovered from the displaced lines L211–215 of the same file.
- **Methodology framing (white paper, not a per-dataset spec):** L133: "the viewing distance and other aspects of the viewing conditions are otherwise kept "plain", that is, representative of 'typical' viewing conditions". L141: "Zooming (2×): Spatially magnifying artefacts that would otherwise remain hidden, in particular on displays with high pixel densities." L170: "While the viewing conditions cannot be fully controlled, crowdsourcing has the advantages of allowing large scale experiments, observer diversity, and a realistic sampling of actual viewing conditions." Note that the white paper (L137) gives flicker as "7.5 Hz", while the DCC paper gives 10 Hz.
- **CfP:** L193 lists "Quality assessment under various viewing conditions" as a use-case category. It gives no subjective geometry.
- **Crop vs full-resolution:** zensim scores both. The labels were collected on crops only (Mohammadi 2025 p11: "Note that only crops were shown").
- **Confidence:** high.

## SDR25 (JPEG-AI-SDR25)

What it is: zensim `DATA_SPLITS.md:235`, "JPEG-AI-SDR25 (5 src × 10 levels, 95k raw triplets) … subjective study behind the QoMEX'25 SVQA paper (arXiv:2504.06301)". It is a subset of AIC-4 ("SDR25 ⊂ AIC-4", same line). Source paper: Jenadeleh et al., "Subjective Visual Quality Assessment for High-Fidelity Learning-Based Image Compression", QoMEX 2025. `P:2b/2bbec4c570709a624d24bbedadff1dfe9abc3ba28a9d674d005a0dc16899f325.pdf`. Repo: `/mnt/v/repos/iqa-tools/jpeg-aic__dataset-JPEG-AI-SDR25/README.md`. The paper's p2 footnote gives the dataset URL: "available at https://github.com/jpeg-aic/dataset-JPEG-AI-SDR25."

- **Setting:** crowd, AMT, reusing the AIC-3 interfaces. p4: "The same two web interfaces developed by JPEG AIC-3 for BTC and PTC were used for this experiment. … Participants were recruited through Amazon Mechanical Turk (MTurk) platform for the BTC and PTC experiments, which were conducted separately."
- **Crop:** p3: "the JPEG AIC group manually selected an interesting region from each source image and cropped it to 620 × 800 pixels. The cropped regions were chosen to retain key structural details and visual complexity, making them representative of the distortions that would be perceived in the full-resolution images."
- **Boosting:** p3: "These include zooming, where the plain images are cropped to half their size and upscaled using Lanczos resampling; artifact amplification, which scales the pixel-wise difference between the original and distorted images by a factor of 2 in each color channel; and flicker effect, where the reference and distorted images alternate at 10 Hz". PTC: "a toggle button allows observers to switch between the compressed and original images … They were also limited to two toggle per seconds."
- **Display, resolution requirement, distance, luminance:** not stated. README L61 and L77 log the per-response `resolution` and per-worker `display_size`.
- **ppd:** not derivable.
- **Confidence:** high.

## UPIQ (Mikhailiuk et al. 2021) and its HDR constituents

Source: "Consolidated Dataset and Metrics for High-Dynamic-Range Image Quality" (arXiv:2012.10758v2) with the supplementary appended. `P:68/6845a362699a223ecea35efda4766ea99851db942438113a3c0e6b9d25a037c1.pdf`. UPIQ merges TID2013, LIVE (SDR), Korshunov [18] and Narwaria [16] (HDR) (p5 Table I). Its native-condition labels come from those studies (see their sections). UPIQ's own contribution is the cross-dataset alignment experiment:

- **Setting:** lab. Supplementary p15: "We ensure that ITU recommendations [6] were met."
- **Displays:** p5–6: "The data necessary for alignment were collected on two different displays. Comparison of SDR to SDR images were performed on a color calibrated 32" SDR Samsung S32D850T display with 2560 × 1440 pixels, 300 cd/m2 typical peak luminance and a black level of ∼0.3 cd/m2. The comparisons involving HDR images were presented on a custom-built, color-calibrated 10" HDR display with 2048 × 1536 pixels, 15,000 cd/m2 peak luminance and a black level below 0.01 cd/m2 [62]."
- **Luminance model for SDR sources:** p6: "Because we had no information on the displays used in the SDR image quality studies, we used the typical parameters of an SDR display: γ = 2.2, the peak luminance, Lpeak = 100 cd/m2, and the black level, Lblack = 0.5 cd/m2. For HDR images, we reproduced the absolute luminance values used in the original studies."
- **Distance / ppd / magnification:** p6: "The viewing distance was 90 cm for both the SDR display (51 pixels per degree) and the HDR display. Both HDR and SDR images, shown on the HDR display, were upscaled by a factor of 3.2 (50 pixels per degree), making the measurements taken for the original SDR and HDR studies comparable with ours." Supplementary p15: "The images were also displayed with the same angular resolution (in pixels per visual degree) as in the original experiments. When the image size exceeded the size of our display, we provided a simple panning interface in which observers could use a trackball to inspect different portion of the image."
- **ppd derived vs stated:** SDR 32" 16:9 → width 0.7084 m, pitch 0.2767 mm, at 0.9 m → **56.8 derived**, against 51 stated. The discrepancy is unexplained; the stated diagonal may not equal the active area. HDR 10" 4:3 → pitch 0.0992 mm, at 0.9 m → 158.3 native; ÷3.2 → **49.5 derived** vs 50 stated (consistent).
- **Note:** "same angular resolution … as in the original experiments" is a claim by the UPIQ authors. Neither TID2013 (free distance) nor CSIQ-style studies give a single ppd, so for TID2013 that angular resolution cannot have come from the TID2013 paper.
- **Confidence:** high for the quotes; medium for SDR ppd (51 vs 56.8).

### Korshunov et al. 2015 (UPIQ HDR constituent)

Source: "Subjective quality assessment database of HDR images compressed with JPEG XT", QoMEX 2015. `P:ea/ea0d07cd3b7e5df4730f4294f8434ed4b3da4bac78884a40ace1372dd23e687c.pdf`.

- **Setting:** lab. p4: "Subjective evaluations were conducted at MMSPG test laboratory, which fulfills the recommendations for subjective evaluation of visual data issued by ITU-R [15]. … the laboratory is equipped with a controlled lighting system with a 6500 K color temperature, a mid gray color is used for all background walls and curtains, and the ambient illumination did not directly reflect off of the monitor. During the experiment, the background luminance behind the monitor was set to 20 lx."
- **Display:** p4: "To display the test stimuli, a full HD 47" SIM2 HDR monitor with individually controlled LED backlight modulation, capable of displaying content with luminance values ranging from 0.001 to 4000 cd/m2, was used. … The red, green, and blue primaries were measured for white set to 1400 cd/m2 level".
- **Distance:** p4: "They were seated in an arc configuration, at a constant distance of 3.2 times the picture height, as suggested in [16]."
- **Crop / scale:** p2: "Images were first cropped and scaled by a factor of two with a bilinear filter to fit their size to 944 × 1080 for side-by-side subjective experiments". p4: "each image was cropped and scaled to 944 × 1080 pixels with 32 pixels of black border separating the two images." Source resolutions ran "from full HD (1920 × 1080) to larger than 4K (6032 × 4018)" (p2).
- **ppd derived:** 1080 rows at 3.2H → **60.3 ppd**.
- **Confidence:** high.

### Narwaria et al. 2013 (UPIQ HDR constituent)

The primary paper (Narwaria, Da Silva, Le Callet, Pépion, "Tone mapping-based high-dynamic-range image compression: study of optimization criterion and perceptual quality", Optical Engineering 52(10), 2013) is **not in the corpus**: manifest title search found nothing, and the ResearchGate fetch returned 403. It is cited in UPIQ as [16] (`6845a362` text L1034). The same authors' HVEI 2014 paper describes the same stimuli set and says for details "The keen reader is also referred to our previous work18" ([18] = the OE 2013 paper). Source (secondary): "On Improving the Pooling in HDR-VDP-2 towards Better HDR Perceptual Quality Assessment", HVEI 2014, `P:cb/cb3d34f0810905b557b2ab8cfe4b1b9c66af6157b3b39397f34e77d36f581242.pdf`.

- **Stimuli match UPIQ's "Narwaria (140, 10 refs)":** p3: "we chose 10 reference (i.e. undistorted) HDR scenes, 7 compression bit rates … we obtained a total of 140 compressed HDR images (10 reference images × 1 TMO × 2 optimization criterion × 7 bit rates)."
- **Setting / display / distance / ambient:** p4: "Observers were seated in a standardized room conforming to the International Telecommunication Union Recommendation (ITU-R) BT500-13 recommendations. For displaying the HDR images, SIM2 HDR47E S 4K display was used. The HDR47E S 4K is a 47-inch, 1080p LCD TV with maximum displayable luminance of 4000 cd/m². The viewing distance was set to three times the height of the screen (active part), that is approximately 178 cm and the room illumination was set to 130cd/m²." (The HVEI paper states "130cd/m²" for room illumination, as printed.) This subjective-test paragraph covers the combined JPEG + JPEG 2000 database. It is not certain that it describes the OE 2013 JPEG session verbatim.
- **Native / crop:** not stated in this source. UPIQ Table I lists Narwaria at 1080×1920.
- **ppd derived:** 1080 rows at 3H → **56.5 ppd**. From the stated 178 cm and a 47" 16:9 active height of 0.585 m → 57.3. The 56.55 figure equals the AIC-4 CTC HDR metric setting.
- **Confidence:** medium (secondary source).

---

## Search log (what was looked for and not found)

- Manifest `~/work/zen/zenpapers/manifest/seed.jsonl`, title/author search: found TID2013, LIVE/Sheikh 2006, MAD/CSIQ, KonJND (TCSVT), KonFiG (IEEE Access 2021), AIC-3 QoMEX'23, AIC-3 DCC, SDR25 QoMEX'25, Mohammadi 2025, UPIQ, Korshunov 2015, CID22 AIC-3 contribution, and dataset pages. **Missing:** KADID-10k QoMEX 2019 paper and Narwaria OE 2013 paper.
- Web: CSIQ official page fetched (quoted above). KADID S2 page returned no content. Narwaria on ResearchGate returned 403. Neither was retrieved.
- `/home/lilith/tmp/zensim-paper/bib/` was listed but not needed; `/mnt/v/repos/iqa-tools/jpeg-aic__*` READMEs were read. No label/MOS/JND files were opened.
