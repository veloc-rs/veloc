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
                // Uncalibrated estimates; memory assumes a cache hit.
                // Sequence costs are aggregate estimates, not instruction counts.
                { class: Address, resource: "Scalar", latency: 1, occupancy: 1 },
                { class: Branch, resource: "Scalar", latency: 1, occupancy: 1 },
                { class: Call, resource: "Scalar", latency: 3, occupancy: 3 },
                { class: FloatAdd, resource: "Scalar", latency: 4, occupancy: 1 },
                { class: FloatCompare, resource: "Scalar", latency: 3, occupancy: 1 },
                { class: FloatDiv, resource: "Scalar", latency: 32, occupancy: 32 },
                { class: FloatMul, resource: "Scalar", latency: 5, occupancy: 1 },
                { class: IntDiv32, resource: "Scalar", latency: 24, occupancy: 24 },
                { class: IntDiv64, resource: "Scalar", latency: 40, occupancy: 40 },
                { class: IntShift, resource: "Scalar", latency: 3, occupancy: 1 },
                { class: Load, resource: "Scalar", latency: 4, occupancy: 1 },
                { class: Push, resource: "Scalar", latency: 4, occupancy: 2 },
                { class: Pop, resource: "Scalar", latency: 4, occupancy: 2 },
                { class: Return, resource: "Scalar", latency: 1, occupancy: 1 },
                { class: Store, resource: "Scalar", latency: 1, occupancy: 1 },
                { class: Trap, resource: "Scalar", latency: 1, occupancy: 1 },
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
