import "features.spec";
import "../schedule.spec";

// Preserve the existing conservative estimates for all three CPU selections.
// This shared model is not a measured Haswell/Skylake port model.
template ScalarModel(Name: ident, Spelling: expr, Features: expr) {
    cpu Name {
        name = Spelling;
        features = Features;
        schedule = {
            issue_width: 1,
            resources: [{ name: "Scalar", units: 1 }],
            classes: [
                { class: Copy, resource: "Scalar", latency: 1, occupancy: 1 },
                { class: IntAlu, resource: "Scalar", latency: 1, occupancy: 1 },
                { class: IntMul, resource: "Scalar", latency: 3, occupancy: 1 },
                { class: IntPopcnt, resource: "Scalar", latency: 3, occupancy: 1 },
                { class: IntToFloat, resource: "Scalar", latency: 4, occupancy: 1 },
                { class: FloatToInt, resource: "Scalar", latency: 4, occupancy: 1 },
                { class: FloatConvert, resource: "Scalar", latency: 4, occupancy: 1 },
                { class: FloatSqrt, resource: "Scalar", latency: 4, occupancy: 1 },
            ],
        };
    }
}

expand ScalarModel(generic, "generic", []);
expand ScalarModel(haswell, "haswell", [SSE41, AVX, AVX2, BMI1, BMI2, POPCNT]);
expand ScalarModel(skylake, "skylake", [SSE41, AVX, AVX2, BMI1, BMI2, POPCNT]);
