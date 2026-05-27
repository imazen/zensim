[v47-recal-negtail] 672 corruption entries vs q20/q10 anchors

[v47-recal-negtail] GATE score(corruption) < score(q20): 116/672 = 17.3% PASS
[v47-recal-negtail] GATE score(corruption) < score(q10): 61/672 = 9.1% PASS

per-family gate(vs q20) pass rate:
  aliasing                             0/ 18    0.0%
  block_copy_wrong                     7/ 18   38.9%
  block_garbage                       10/ 18   55.6%
  block_gray                           5/ 18   27.8%
  block_repeat_neighbor                3/ 18   16.7%
  block_zero                           6/ 18   33.3%
  channel_invert                       9/ 18   50.0%
  channel_max_r                        6/ 18   33.3%
  channel_swap_gb                      3/ 18   16.7%
  channel_swap_rb                      0/ 18    0.0%
  channel_swap_rg                      1/ 18    5.6%
  channel_zero_b                       0/ 18    0.0%
  channel_zero_g                       8/ 18   44.4%
  channel_zero_r                       1/ 18    5.6%
  chroma_boundary                      0/ 18    0.0%
  composite_premul_as_straight         8/ 18   44.4%
  composite_wrong_bg_black             3/ 18   16.7%
  composite_wrong_bg_white             6/ 18   33.3%
  edge_border_all_k1                   0/  3    0.0%
  edge_border_all_k2                   1/  3   33.3%
  edge_border_all_k4                   2/  3   66.7%
  edge_border_top_k1                   0/  3    0.0%
  edge_border_top_k2                   0/  3    0.0%
  edge_border_top_k4                   0/  3    0.0%
  edge_duplicate_top_row               0/  3    0.0%
  edge_shift_interior1px               0/  3    0.0%
  geometric_flip_h                     6/ 18   33.3%
  geometric_flip_v                     6/ 18   33.3%
  geometric_rotate90                   7/ 18   38.9%
  geometric_shift1px                   0/ 18    0.0%
  noise_bit_flip_n1                    0/ 18    0.0%
  noise_bit_flip_n16                   0/ 18    0.0%
  noise_bit_flip_n256                  0/ 18    0.0%
  noise_salt_pepper_n1                 0/ 18    0.0%
  noise_salt_pepper_n16                0/ 18    0.0%
  noise_salt_pepper_n256               0/ 18    0.0%
  overlay_glyph                        0/ 18    0.0%
  overlay_line                         0/ 18    0.0%
  overlay_rect                         8/ 18   44.4%
  tone_brightness_d40neg               1/ 18    5.6%
  tone_brightness_d40pos               0/ 18    0.0%
  tone_contrast_boost                  3/ 18   16.7%
  tone_gamma_decode                    4/ 18   22.2%
  tone_gamma_encode                    2/ 18   11.1%

per-region gate(vs q20) pass rate (smaller = harder/subtler):
  whole     47/132   35.6%
  frac2     37/108   34.3%
  frac4     23/108   21.3%
  sq64       9/108    8.3%
  sq16       0/108    0.0%
  sq8        0/108    0.0%

556 FAILURES (corruption >= q20). worst 20:
  name                                                     corr      q20
  gb82_dog__block_repeat_neighbor__whole__op100            97.8     43.9
  gb82_dog__block_repeat_neighbor__whole__op20             97.8     43.9
  gb82_dog__block_repeat_neighbor__whole__op50             97.8     43.9
  gb82_dog__noise_bit_flip_n1__whole__op20                 97.2     43.9
  gb82_dog__noise_bit_flip_n1__frac4__op20                 97.1     43.9
  gb82_dog__noise_bit_flip_n1__whole__op50                 97.1     43.9
  gb82_dog__noise_bit_flip_n1__whole__op100                97.0     43.9
  gb82_dog__chroma_boundary__sq8__op20                     97.0     43.9
  gb82_dog__noise_bit_flip_n1__sq64__op20                  97.0     43.9
  gb82_dog__noise_bit_flip_n1__frac4__op50                 97.0     43.9
  gb82_dog__channel_swap_rb__sq8__op20                     96.9     43.9
  gb82_dog__noise_bit_flip_n1__sq16__op50                  96.8     43.9
  gb82_dog__chroma_boundary__sq16__op20                    96.8     43.9
  gb82_dog__noise_bit_flip_n1__frac4__op100                96.7     43.9
  gb82_dog__noise_bit_flip_n1__sq64__op50                  96.7     43.9
  gb82_dog__noise_bit_flip_n1__sq16__op20                  96.7     43.9
  gb82_dog__chroma_boundary__sq8__op50                     96.7     43.9
  gb82_dog__channel_swap_rb__sq16__op20                    96.7     43.9
  gb82_dog__channel_swap_rb__sq8__op50                     96.6     43.9
  gb82_dog__noise_bit_flip_n1__frac2__op20                 96.6     43.9
