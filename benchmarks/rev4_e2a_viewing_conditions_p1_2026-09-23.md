# D3: viewing conditions of the human IQA datasets (lane E2a, zensim Rev4)

Read-only research, 2026-09-23. Every claim below is a verbatim quote with a locator. Where a source says nothing, the field reads **not stated**. Pixels-per-degree values marked **derived** are my arithmetic, done with the per-pixel formula

    ppd = 1 / deg( 2 * atan( 0.5 * pixel_pitch / distance ) )

which is the formula pycvvdp uses. I checked it against the presets: it reproduces `standard_fhd` = 37.84 and `standard_4k` = 75.40. When the distance is given in screen/picture heights (kH) and H spans N pixel rows, pitch = H/N, so ppd depends only on N and k.

Paper locators: `P:` = `/mnt/v/input/papers/`. "pN" = PDF page N as extracted by `pdftotext -layout` (form-feed page index). Line numbers are given where a `.md` file was quoted. PDF text comes from two-column layouts, so quotes join column fragments in reading order. Hyphenated line breaks are kept as printed.


> Deliverable 3 of lane E2a (zensim Rev4, 2026-09-23). Verbatim passages from each dataset's own paper, gathered by a read-only helper agent and spot-checked by the lane (CID22 0.0213°, KonJND 30 cm, UPIQ 51 ppd, Korshunov 3.2 H re-found in the sources). Split in two parts to respect the 30 KB rule; full original: `/var/tmp/rev4-e2a/d3_viewing_conditions.md`. (part 1 of 2: summary, CID22 … KonFiG-IQA)

## Summary table

| dataset | setting | display | resolution (screen) | distance | ppd (stated / **derived**) | nearest CVVDP preset |
|---|---|---|---|---|---|---|
| CID22 | crowd (Subjectify), desktop/laptop only | not stated (workers' own) | not stated | not stated | DSBQS/MCOS: stated 0.0213°/CSS px → **46.9 derived**; TSBPC: upscaled to fill screen height → not derivable | between standard_fhd (37.84) and sdr_4k_30 (60.55); no exact preset |
| KonJND-1k | crowd (AMT), desktop/laptop, Chromium | workers' own, ≥13.3 in, credit-card calibrated | logical ≥1366×768 | **30 cm** (requested) | **24.3 derived** (640 px shown at 13.797 cm, at 30 cm) | below standard_fhd (37.84). None this low, so needs a custom geometry |
| KADID-10k | crowd (figure-eight.com) | not stated ("variable screen resolutions") | not stated | not stated | not derivable | none justified; standard_fhd is only a default |
| TID2013 | mixed lab (tutored) + Internet, 5 countries | "LCD and CRT, mainly 19'' and more" | "1152 [×] 864" | "comfortable for them" | not derivable (distance free) | none justified |
| CSIQ | lab | "four calibrated LCD monitors placed side by side" | not stated | "equal viewing distance" (value not stated) | not derivable | none justified |
| LIVE (R2) | lab (office) | 21-inch CRT, uncalibrated | 1024×768 | 2–2.5 screen heights | **26.8–33.5 derived** | nearest standard_fhd (37.84), still above the LIVE range |
| KonFiG-IQA | crowd (AMT) | not stated | not stated | not stated | not derivable. Stimuli pre-zoomed 2× for the Z/AZ/ZF boosts | none justified |
| AIC-3 dataset (Testolina QoMEX'23, 10 refs) | crowd (QualityCrowd 2), expert viewers | not stated | screen ≥1920×1080, DPR = 1 | not stated | not derivable | none justified. DPR=1 on ≥FHD suggests desktop, but the distance is unknown |
| AIC-3 BTC/PTC study (Testolina DCC'25, 5 refs; = AIC-4 sample stimuli) | crowd (AMT) | not stated | not stated (per-response `resolution` + `display_size` logged) | not stated | not derivable. BTC = 2× zoom + 2× amplification + flicker; PTC = plain, in-place toggle | none justified |
| AIC-4 sample / AIC-4 CTC | same study as above (CTC gives no subjective viewing geometry) | — | — | — | CTC gives **metric configuration** only: SDR 37.84 ppd / 200 or 203 cd/m²; HDR 56.55 ppd / 1000 cd/m² (**stated, for anchor metrics, not the human study**) | standard_fhd, matching the CTC's own CVVDP anchor setting |
| SDR25 (JPEG-AI-SDR25) | crowd (AMT), JPEG AIC-3 web interfaces | not stated | not stated | not stated | not derivable; same BTC/PTC boosting as AIC-3 | same as AIC-4 (SDR25 ⊂ AIC-4 per zensim DATA_SPLITS) |
| UPIQ (alignment expts) | lab | SDR: 32" Samsung S32D850T, 2560×1440, 300 cd/m²; HDR: custom 10" 2048×1536, 15,000 cd/m² | as left | 90 cm | stated SDR 51 ppd (my derivation from 32"/2560 px/90 cm gives **56.8**, so they disagree); HDR stated 50 ppd after ×3.2 upscale (**49.5 derived**) | SDR ≈ between standard_fhd and sdr_4k_30; the table below explains why no preset has 51 |
| └ Korshunov 2015 (UPIQ HDR) | lab (EPFL MMSPG, BT.500) | 47" SIM2 HDR, full HD, 0.001–4000 cd/m², 20 lx background | 1920×1080 (two 944×1080 crops side by side) | 3.2 picture heights | **60.3 derived** | sdr_4k_30 geometry (60.55) is closest in ppd. Luminance is HDR, so no SDR preset fits; needs a custom HDR preset |
| └ Narwaria 2013 (UPIQ HDR) | lab (BT.500 room) — **per the authors' HVEI 2014 paper; OE 2013 paper not retrieved** | SIM2 HDR47E S 4K, 47-inch 1080p, 4000 cd/m² | 1080p | 3H ≈ 178 cm | **56.5 derived** (3H) / 57.3 (178 cm) — equals the AIC-4 CTC HDR 56.55 | custom HDR preset; ppd ≈ 56.5 |

Preset geometry (from `/home/lilith/work/zen/zenmetrics/crates/cvvdp/data/display_models.json`, derived with the formula above): `standard_4k` 30" 3840×2160 @ 0.7472 m → **75.40**; `standard_fhd` 24" 1920×1080 @ 0.6 m → **37.84**; `sdr_fhd_24` identical geometry → 37.84 (100 cd/m² peak vs 200); `sdr_4k_30` 30" 3840×2160 @ 0.6 m → **60.55** (100 cd/m²); `standard_phone` 6" 2400×1080 @ 0.4 m → **120.6**; `iphone_14_pro` 6.1" 2532×1170 @ 20 in → **159.6**; `eizo_CG3146` → 75.19; `macbook_pro_16` → 98.8. So the brief's "standard_4k = 75.4 ppd, 4K 30in" is confirmed, and `standard_fhd` = 37.84 is confirmed. No shipped preset lands in the 24–35 ppd band that KonJND, LIVE and (probably) TID2013 occupy. None lands at 47 (CID22 DSBQS), 51 (UPIQ SDR stated) or 56.5 (Narwaria/CTC-HDR) either.

---

## CID22

Source: "AIC-3 Contribution from Cloudinary: CID22" (Sneyers, Ben Baruch, Vaxman). `P:5b/5b69d93b4bb7bb02f3f1bb76322788ffa526c6064b0d0f1488f399a0bc6c81b1.pdf` (md alongside).

- **Setting:** crowdsourced, platform Subjectify, desktop/laptop only.
  - p7: "They were instructed to use a desktop or laptop for the experiment, and this was checked during recruitment." / "The crowd-sourcing platform Subjectify was used to perform this experiment." / screening rule 3: "the participant had switched to a mobile device (phone or tablet) between recruitment and actually performing the test, despite the instruction to use a desktop or laptop."
- **Display model / resolution:** not stated (workers' own devices).
- **Presentation, DSBQS (absolute MOS, the anchor protocol whose results take precedence in MCOS):** p4 right column: "The images are not displayed with scaling to fit the screen, but at 'dpr1' resolution, i.e. how the image gets displayed by default in a web browser when the image is in a simple <img> tag without additional layout — in case of normal-density screens, this means one image pixel corresponds to one display pixel (1:1); in case of high-density ('retina') screens, this means one image pixel corresponds to 2x2 display pixels (2:1). In other words, one image pixel corresponds to one CSS pixel[15], which theoretically corresponds to a visual angle of 0.0213 degrees (though in practice this may only be an approximation). The aim is to make the viewing conditions as uniform as possible between test subjects — although in a crowd-sourced setup, large differences in viewing conditions will inevitably remain."
  - p5: "In this protocol, while "manual flickering" is still allowed, the images are displayed without additional upscaling (only adjustment for high-density displays), in order to make the conditions more consistent across participants and to limit the visibility of artifacts to a relevant level."
- **Presentation, TSBPC (pairwise RMOS):** p4: "Additionally, the images are displayed with upscaling in order to fill the screen height minus the space needed for the interface." / "The images are scaled up to ensure that the physical dimensions are large enough also on high density displays".
- **Image size:** p5: "All the images have pixel dimensions of 512×512."
- **Viewing distance:** not stated. p15: "In the DSBQS experiment, we did not attempt to model the effect of viewing conditions. We tried to make the conditions as uniform as possible by only allowing the use of desktop or laptop computers (not mobile phones) and by displaying images at similar angular dimensions (to the extent that this can be done in a crowd-sourced experiment)."
- **Luminance / ambient:** not stated. p15: "We only used images in the sRGB color space."
- **ppd:** the stated angle is 0.0213°/image px (DSBQS), so ppd = 1/0.0213 = **46.9 (derived from the stated angle)**. The 0.0213° figure is the CSS reference-pixel value the authors cite, and they flag it as "only an approximation". TSBPC magnification depends on screen height, so its ppd is not derivable.
- **Confidence:** high for protocol text; the 46.9 ppd is a nominal figure, not a measurement.

## KonJND-1k

Source: Lin et al., "Large-scale crowdsourced subjective assessment of picturewise just noticeable difference", IEEE TCSVT 2022. `P:c2/c20a173c2bcff17d83a8fc165ec00bdbce5797b9bbd02cc9bba97fc4905aa5f7.pdf`.

- **Setting:** crowd, AMT. p3: "samples by 503 crowd workers is collected via Amazon Mechanical Turk (AMT)".
- **Device requirements:** p7 right column: "• Desktops and laptops are allowed, while mobile phones and tablets are not. • Use of a Chromium-based browser such as Google Chrome. • The display screen must have a minimum logical resolution of 1366 × 768."
- **Physical size calibration:** p7: "to align the viewing conditions, we displayed all images at the same physical size and fixed the viewing distance for all workers. To display the entire graphical user interface (GUI) of our subjective experiment, workers were required to have a minimum display size of 13.3 inches." … "Workers were asked to prepare a credit card with a size of 85.60 mm × 53.98 mm or a card of the same size and adjust the size of a frame on the screen until the frame fits the card size (Fig. 4)." … "Displaying an image with a resolution of 640 × 480 on a screen with a logical resolution of 1366 × 768 and a physical size of 13.3 inches requires a physical dimension of 13.797 cm × 10.347 cm. We displayed our test images on all workers' screens in this physical size."
- **Zoom lock:** p8: "Their browsers were blocked whenever they changed the browser zoom level after calibration."
- **Viewing distance:** p8: "After the calibration was finished, we asked workers to adjust their viewing distance to 30 cm. The suggested viewing distance was derived according to trigonometric calculation [33], [34] and ISO standard [35]." (The distance is requested, not enforced.)
- **Image size:** p5: images "were then scaled and cropped to a size of 640 × 480 pixels"; p6: "This choice of image resolution was motivated by the layout of the user interface in our experiments and the minimum required screen resolution".
- **Presentation:** flicker at 8 Hz. p6: "We used a flicker test, with the reference and the compressed test image being displayed successively at a frequency of 8 Hz."
- **Luminance / ambient:** not stated. The paper itself notes (p3) that "The results depend, for example, on display size, viewing distance, environmental conditions".
- **ppd derived:** pitch = 13.797 cm / 640 = 0.021558 cm; at 30 cm → **24.3 ppd**. The image spans 25.9° horizontally.
- **Confidence:** high. This is the best-controlled crowd geometry in the set.

## KADID-10k

The primary paper (Lin, Hosu, Saupe, "KADID-10k: A Large-scale Artificially Distorted IQA Database", QoMEX 2019) is **not in the corpus**. I searched the manifest titles and grepped for "figure-eight". A web search found only the Semantic Scholar/Scribd/PUMA listings, and the S2 page fetch returned no content. I used the same authors' DeepFL-IQA paper (IEEE TMM 2020), which describes the same KADID-10k crowd study: `P:fd/fd90bdd6062bdd5f464e3e8b409e8ebeec6cea8336f38e0147c7622b33526f43.pdf`. The dataset page is `P:c6/c6990288…5711616.md` and has no viewing info.

- **Setting:** crowd. p5: "Conducting a subjective lab study on 10,125 images is time-consuming and costly; instead, we performed the study on figure-eight.com, a crowdsourcing platform."
- **Conditions:** p5: "Due to the weakly controlled nature of the experimental setup, including factors such as variable screen resolutions and screen contrast, ambient illumination, etc., crowdsourced subjective studies are less reliable than lab-based ones. We used a standard degradation category rating (DCR) method [53] to reduce these effects."
- **Layout:** p5: "given the pristine image on the left side, the crowd workers were asked to rate the distorted image on the right side in relation to the pristine reference image on a 5-point scale".
- **Image size:** p4: images "were rescaled to span … the same resolution as that in TID2013 (512×384)".
- **Display, resolution, distance, luminance, magnification:** not stated.
- **ppd:** not derivable.
- **Confidence:** medium. The source is secondary (same authors, same study); the QoMEX 2019 paper was not read.

## TID2013

Source: Ponomarenko et al., "Image database TID2013: Peculiarities, results and perspectives", Signal Processing: Image Communication 30 (2015). `P:d3/d3593f970ca6002114381deba9e196deea61d996a788332fb9c60ef25de0c9e5.pdf`.

- **Setting:** mixed lab and Internet. p11–12: "Experiments for TID2013 were conducted in five countries (Finland, France, Italy, Ukraine, USA). … it was possible to carry out experiments both in laboratory conditions (under control of tutors) and distantly via Internet."
- **Display / resolution / distance:** p12: "subjective tests have been done in different conditions. In particular, different monitors were used, both LCD and CRT, mainly 19'' and more with the resolution 1152 [×] 864 pixel. More than 300 observers have performed experiments via Internet. … Observation conditions varied in reasonable limits and we asked participants to use distance from monitors comfortable for them. All these do not correspond to stricter requirements imposed by ITU." (The "×" is dropped in the PDF text layer.)
  - p11: "For experiments carried out via Internet, a participant had to read Instructions related to preferred (recommended) conditions and a methodology of experiments. In particular, it was recommended to use convenient (preferred) distance to a monitor".
- **Image size / layout:** p2: "Two or three images are usually displayed simultaneously at the monitor screen … For both TID2008 and TID2013 it was supposed that images were displayed at computer monitors. Because of this, all images were of the same fixed size 512 [×] 384 pixel".
- **Luminance / ambient:** not stated.
- **ppd:** not derivable (distance self-chosen).
- **Confidence:** high that conditions were uncontrolled.

## CSIQ

Sources: Larson & Chandler, "Most apparent distortion…", J. Electronic Imaging 19(1) 2010. `P:92/92495fc454e2cc49fa5803b6a9cb6d6e119ec7a983cb62cff309ac8756013836.pdf`. QUALINET page `P:cd/cdc7a854….md`. Official CSIQ page https://s2.smu.edu/~eclarson/csiq.html (web source).

- **Setting:** lab (calibrated monitor array).
- **Display:** p10 (right column): "CSIQ images are subjectively rated based on a linear displacement of the images across four calibrated LCD monitors placed side by side with equal viewing distance to the observer. All of the distorted versions of an original image were viewed simultaneously on the monitor array and placed in relation to one another according to overall quality."
  - Web source (s2.smu.edu CSIQ page), same sentence: "CSIQ images are subjectively rated base on a linear displacement of the images across four calibrated LCD monitors placed side by side with equal viewing distance to the observer." The page gives no model, resolution or distance.
- **Monitor model, resolution, distance value, luminance, ambient:** not stated.
- **ppd:** not derivable.
- **Confidence:** high that nothing more is stated in these sources. The CSIQ subjective protocol may be described elsewhere (e.g. a Chandler thesis or tech report), but I did not find it in the corpus.

## LIVE (Release 2; Sheikh, Sabir, Bovik 2006)

Source: "A Statistical Evaluation of Recent Full Reference Image Quality Assessment Algorithms", IEEE TIP 2006. `P:ab/abae95ccc661e20004e4962a3bc86697cc39b4f2aee19df412c36df872abe20c.pdf`. The LIVE web page `P:85/85e43c04….md` has no viewing details.

- **Setting:** lab, office. p5–6: "The experiments were conducted using identical Microsoft Windows workstations. A web-based interface showing the image to be ranked and a Java scale-and-slider applet for assigning a quality score was used. The workstations were placed in an office environment with normal indoor illumination levels."
- **Display / resolution / distance:** p6: "The display monitors were all 21-inch CRT monitors displaying at a resolution of 1024 × 768 pixels. Although the monitors were not calibrated, they were all approximately the same age, and set to the same display settings. Subjects viewed the monitors from an approximate viewing distance of 2-2.5 screen heights."
- **Image size:** p4: "These images were resized (using bicubic interpolation) to a reasonable size for display on a screen resolution of 1024 × 768 that we had chosen for the experiments. Most images were 768 × 512 pixels in size."
- **Luminance:** not stated (monitors uncalibrated). Ambient: "normal indoor illumination levels".
- **ppd derived:** H = 768 rows; at 2H → **26.8 ppd**; at 2.5H → **33.5 ppd**. This uses the stated resolution and assumes the active raster fills the screen height the authors measured distance by.
- **Confidence:** high.

## KonFiG-IQA

Source: Men, Lin, Jenadeleh, Saupe, "Subjective Image Quality Assessment With Boosted Triplet Comparisons", IEEE Access 2021. `P:45/45ae66c956b6e105b90591ea6e1b073f78813c388ce8dd9dbc17fc4b399ae6ac.pdf`. The paper introduces "KonFiG-IQA (Konstanz Fine-Grained IQA)" (p3).

- **Setting:** crowd, AMT. p16: "The experiments were carried out on the Amazon Mechanical Turk [66] platform". Table p4: "KonFiG-IQA (Part A) 2021 … Crowdsourcing".
- **Image size / magnification:** p19: "Ten source images were selected from the MCL-JCI dataset [29], whose original resolution is 1080 × 1920. In our subjective study, the original resolution is too large to display on the screens of crowd workers. To ensure that a triplet can be displayed without image re-scaling, we manually cropped each image to 512×384 pixels. … We further cropped the images to 256×196 pixels for experiments with boosting by zooming and subsequently upscaled them back to 512×384 pixels for display." (The Fig. 10 caption gives the zoom inset as 256 × 192.)
  - p12: "participants in an IQA experiment may be tempted to enlarge the images displayed in their browser or to move closer to the screen to detect fine differences between images. However, to ensure a uniform and controlled quality assessment, participants are asked to refrain from such adhoc zooming action. Instead, we propose to deliver the displayed images already in a zoomed and cropped fashion." / "Images are cropped to half their linear size and zoomed by a factor of two. … bicubic interpolation was adopted for scaling up." / "In this paper, we chose a fixed zoom factor of two."
  - Flicker: p12: "a distorted image and its reference are displayed successively at a frequency of 8 Hz."
- **Display, screen resolution requirement, distance, luminance:** not stated. I grepped for screen, desktop, laptop, browser, distance and resolution. The only related hit is p17's "workers' screens and devices" in a list of error sources.
- **ppd:** not derivable. Note that the Z/AZ/ZF/AZF boosted conditions are 2× magnified relative to the plain ones.
- **Confidence:** high that the geometry is unstated.

