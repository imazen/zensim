[qat-native-TILEMIN] 672 corruption entries vs q20/q10 anchors

[qat-native-TILEMIN] GATE score(corruption) < score(q20): 248/672 = 36.9% PASS
[qat-native-TILEMIN] GATE score(corruption) < score(q10): 123/672 = 18.3% PASS

per-family gate(vs q20) pass rate:
  aliasing                             0/ 18    0.0%
  block_copy_wrong                    10/ 18   55.6%
  block_garbage                       15/ 18   83.3%
  block_gray                          10/ 18   55.6%
  block_repeat_neighbor                8/ 18   44.4%
  block_zero                          11/ 18   61.1%
  channel_invert                      16/ 18   88.9%
  channel_max_r                        9/ 18   50.0%
  channel_swap_gb                      9/ 18   50.0%
  channel_swap_rb                      6/ 18   33.3%
  channel_swap_rg                      8/ 18   44.4%
  channel_zero_b                       0/ 18    0.0%
  channel_zero_g                      15/ 18   83.3%
  channel_zero_r                       6/ 18   33.3%
  chroma_boundary                      0/ 18    0.0%
  composite_premul_as_straight        15/ 18   83.3%
  composite_wrong_bg_black             9/ 18   50.0%
  composite_wrong_bg_white            10/ 18   55.6%
  edge_border_all_k1                   1/  3   33.3%
  edge_border_all_k2                   1/  3   33.3%
  edge_border_all_k4                   2/  3   66.7%
  edge_border_top_k1                   0/  3    0.0%
  edge_border_top_k2                   1/  3   33.3%
  edge_border_top_k4                   2/  3   66.7%
  edge_duplicate_top_row               0/  3    0.0%
  edge_shift_interior1px               1/  3   33.3%
  geometric_flip_h                    10/ 18   55.6%
  geometric_flip_v                    11/ 18   61.1%
  geometric_rotate90                  11/ 18   61.1%
  geometric_shift1px                   1/ 18    5.6%
  noise_bit_flip_n1                    0/ 18    0.0%
  noise_bit_flip_n16                   0/ 18    0.0%
  noise_bit_flip_n256                  2/ 18   11.1%
  noise_salt_pepper_n1                 0/ 18    0.0%
  noise_salt_pepper_n16                0/ 18    0.0%
  noise_salt_pepper_n256               3/ 18   16.7%
  overlay_glyph                        4/ 18   22.2%
  overlay_line                         1/ 18    5.6%
  overlay_rect                        14/ 18   77.8%
  tone_brightness_d40neg               4/ 18   22.2%
  tone_brightness_d40pos               2/ 18   11.1%
  tone_contrast_boost                  6/ 18   33.3%
  tone_gamma_decode                    9/ 18   50.0%
  tone_gamma_encode                    5/ 18   27.8%

per-region gate(vs q20) pass rate (smaller = harder/subtler):
  whole     62/132   47.0%
  frac2     56/108   51.9%
  frac4     53/108   49.1%
  sq64      46/108   42.6%
  sq16      19/108   17.6%
  sq8       12/108   11.1%

424 FAILURES (corruption >= q20). worst 20:
  name                                                     corr      q20
  gb82_dog__block_repeat_neighbor__whole__op100            97.7     31.1
  gb82_dog__block_repeat_neighbor__whole__op20             97.7     31.1
  gb82_dog__block_repeat_neighbor__whole__op50             97.7     31.1
  gb82_dog__chroma_boundary__sq8__op20                     96.2     31.1
  gb82_dog__noise_bit_flip_n1__frac4__op20                 96.1     31.1
  gb82_dog__noise_bit_flip_n1__frac2__op20                 95.8     31.1
  gb82_dog__noise_bit_flip_n1__whole__op20                 95.8     31.1
  gb82_dog__noise_bit_flip_n1__sq16__op20                  95.8     31.1
  gb82_dog__chroma_boundary__sq16__op20                    95.8     31.1
  gb82_dog__noise_bit_flip_n1__sq64__op20                  95.7     31.1
  gb82_dog__noise_bit_flip_n1__frac4__op50                 95.7     31.1
  gb82_dog__noise_bit_flip_n1__sq8__op20                   95.7     31.1
  gb82_dog__channel_swap_rb__sq8__op20                     95.7     31.1
  gb82_dog__noise_bit_flip_n1__sq16__op50                  95.7     31.1
  gb82_dog__chroma_boundary__sq8__op50                     95.6     31.1
  gb82_dog__noise_bit_flip_n1__frac2__op50                 95.4     31.1
  gb82_dog__channel_swap_rb__sq16__op20                    95.4     31.1
  gb82_dog__channel_swap_rb__sq8__op50                     95.3     31.1
  gb82_dog__noise_bit_flip_n1__sq16__op100                 95.3     31.1
  gb82_dog__noise_bit_flip_n1__sq8__op50                   95.2     31.1
