#[path = "../../tests/common/generators.rs"]
mod generators;

pub fn pair(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let src = generators::gen_checkerboard(w, h, 13);
    let mut dst = src.clone();
    for (i, p) in dst.iter_mut().enumerate() {
        p[i % 3] = p[i % 3].saturating_add((i % 11 + 1) as u8);
    }
    (src, dst)
}
