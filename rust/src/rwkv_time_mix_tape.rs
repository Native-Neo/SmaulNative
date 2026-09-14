use ndarray::{Array1, Array2, Array4};

pub struct RwkvTimeMixTape {
    pub input: Array2<f32>,
    pub prev: Array1<f32>,
    pub xr: Array2<f32>,
    pub xw: Array2<f32>,
    pub xk: Array2<f32>,
    pub xv: Array2<f32>,
    pub xa: Array2<f32>,
    pub xg: Array2<f32>,
    pub r: Array2<f32>,
    pub g_decay: Array2<f32>,
    pub k: Array2<f32>,
    pub v: Array2<f32>,
    pub a: Array2<f32>,
    pub g: Array2<f32>,
    pub kk: Array2<f32>,
    pub k_mod: Array2<f32>,
    pub w: Array2<f32>,
    pub state_initial: Array4<f32>,
    pub y: Array2<f32>,
    pub normalized: Array2<f32>,
    pub output: Array2<f32>,
    pub v_first: Array2<f32>,
}

impl RwkvTimeMixTape {
    pub fn new(input: Array2<f32>, prev: Array1<f32>, state_initial: Array4<f32>) -> Self {
        let shape = input.raw_dim();
        let channels = input.ncols();
        let state_shape = state_initial.raw_dim();
        Self {
            input,
            prev,
            xr: Array2::zeros(shape.clone()),
            xw: Array2::zeros(shape.clone()),
            xk: Array2::zeros(shape.clone()),
            xv: Array2::zeros(shape.clone()),
            xa: Array2::zeros(shape.clone()),
            xg: Array2::zeros(shape.clone()),
            r: Array2::zeros(shape.clone()),
            g_decay: Array2::zeros(shape.clone()),
            k: Array2::zeros(shape.clone()),
            v: Array2::zeros(shape.clone()),
            a: Array2::zeros(shape.clone()),
            g: Array2::zeros(shape.clone()),
            kk: Array2::zeros(shape.clone()),
            k_mod: Array2::zeros(shape.clone()),
            w: Array2::zeros(shape),
            state_initial: Array4::zeros(state_shape),
            y: Array2::zeros((input.nrows(), channels)),
            normalized: Array2::zeros((input.nrows(), channels)),
            output: Array2::zeros((input.nrows(), channels)),
            v_first: Array2::zeros((input.nrows(), channels)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, Array4};

    #[test]
    fn tape_preserves_forward_shapes() {
        let tape = RwkvTimeMixTape::new(
            Array2::zeros((3, 8)),
            Array1::zeros(8),
            Array4::zeros((1, 2, 4, 4)),
        );
        assert_eq!(tape.xr.dim(), (3, 8));
        assert_eq!(tape.state_initial.dim(), (1, 2, 4, 4));
        assert_eq!(tape.output.dim(), (3, 8));
    }
}
