use ndarray::{Array1, Array2, Array4};

pub struct RwkvBlockTape {
    pub input: Array2<f32>,
    pub ln0_output: Option<Array2<f32>>,
    pub ln1_output: Array2<f32>,
    pub time_output: Array2<f32>,
    pub residual: Array2<f32>,
    pub ln2_output: Array2<f32>,
    pub cmix_output: Array2<f32>,
    pub output: Array2<f32>,
    pub time_state: Array4<f32>,
    pub time_prev: Array1<f32>,
    pub cmix_prev: Array1<f32>,
}

impl RwkvBlockTape {
    pub fn new(input: Array2<f32>, output: Array2<f32>) -> Self {
        let rows = input.nrows();
        let channels = input.ncols();
        Self {
            input,
            ln0_output: None,
            ln1_output: Array2::zeros((rows, channels)),
            time_output: Array2::zeros((rows, channels)),
            residual: Array2::zeros((rows, channels)),
            ln2_output: Array2::zeros((rows, channels)),
            cmix_output: Array2::zeros((rows, channels)),
            output,
            time_state: Array4::zeros((1, 1, 1, 1)),
            time_prev: Array1::zeros(channels),
            cmix_prev: Array1::zeros(channels),
        }
    }

    pub fn clear(&mut self) {
        self.ln0_output = None;
        self.ln1_output.fill(0.0);
        self.time_output.fill(0.0);
        self.residual.fill(0.0);
        self.ln2_output.fill(0.0);
        self.cmix_output.fill(0.0);
        self.time_state.fill(0.0);
        self.time_prev.fill(0.0);
        self.cmix_prev.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tape_has_all_residual_stages() {
        let tape = RwkvBlockTape::new(Array2::zeros((2, 4)), Array2::zeros((2, 4)));
        assert_eq!(tape.ln1_output.dim(), (2, 4));
        assert_eq!(tape.residual.dim(), (2, 4));
        assert_eq!(tape.cmix_output.dim(), (2, 4));
    }
}
