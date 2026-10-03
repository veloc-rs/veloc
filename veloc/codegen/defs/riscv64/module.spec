import "abi.spec";
import "features.spec";
import "schedule.spec";
import "lower.spec";
// Even the generic model explicitly owns its estimates and execution resources.
cpu generic {
    name = "generic";
    features = [I, M, F, D, C];
    schedule = {
        issue_width: 1,
        resources: [{ name: "Scalar", units: 1 }],
        classes: [
            // Uncalibrated estimates; memory assumes a cache hit.
            // Sequence costs are aggregate estimates, not instruction counts.
            { class: Address, resource: "Scalar", latency: 8, occupancy: 8 },
            { class: Branch, resource: "Scalar", latency: 1, occupancy: 1 },
            { class: Call, resource: "Scalar", latency: 3, occupancy: 3 },
            { class: Constant32, resource: "Scalar", latency: 2, occupancy: 2 },
            { class: Constant64, resource: "Scalar", latency: 8, occupancy: 8 },
            { class: Copy, resource: "Scalar", latency: 1, occupancy: 1 },
            { class: FloatAdd, resource: "Scalar", latency: 4, occupancy: 1 },
            { class: FloatCompare, resource: "Scalar", latency: 3, occupancy: 1 },
            { class: FloatComparePair, resource: "Scalar", latency: 4, occupancy: 2 },
            { class: FloatConvert, resource: "Scalar", latency: 4, occupancy: 1 },
            { class: FloatDiv, resource: "Scalar", latency: 32, occupancy: 32 },
            { class: FloatMul, resource: "Scalar", latency: 5, occupancy: 1 },
            { class: FloatSqrt, resource: "Scalar", latency: 32, occupancy: 32 },
            { class: FloatToInt, resource: "Scalar", latency: 4, occupancy: 1 },
            { class: IntPair, resource: "Scalar", latency: 2, occupancy: 2 },
            { class: IntToFloat, resource: "Scalar", latency: 4, occupancy: 1 },
            { class: Load, resource: "Scalar", latency: 4, occupancy: 1 },
            { class: Return, resource: "Scalar", latency: 1, occupancy: 1 },
            { class: Select, resource: "Scalar", latency: 4, occupancy: 4 },
            { class: Store, resource: "Scalar", latency: 1, occupancy: 1 },
            { class: Trap, resource: "Scalar", latency: 1, occupancy: 1 },
            { class: IntAlu, resource: "Scalar", latency: 1, occupancy: 1 },
            { class: IntMul32, resource: "Scalar", latency: 3, occupancy: 1 },
            { class: IntMul64, resource: "Scalar", latency: 4, occupancy: 1 },
            { class: IntDiv32, resource: "Scalar", latency: 20, occupancy: 1 },
            { class: IntDiv64, resource: "Scalar", latency: 20, occupancy: 1 },
        ],
    };
}
cpu c908 {
    name = "c908";
    features = [I, M, F, D, C, Zba, Zbb];
    // Integer costs calibrated on K230; added classes below are uncalibrated.
    schedule = {
        issue_width: 2,
        resources: [
            { name: "Alu", units: 2 },
            { name: "Mul", units: 1 },
            { name: "Div", units: 1 },
            // Shared fallback pool until the additional units are calibrated.
            { name: "Uncalibrated", units: 1 },
            { name: "LoadStore", units: 1 },
        ],
        classes: [
            // Uncalibrated estimates; memory assumes a cache hit.
            // Sequence costs are aggregate estimates, not instruction counts.
            { class: Address, resource: "Alu", latency: 8, occupancy: 8 },
            { class: Branch, resource: "Uncalibrated", latency: 1, occupancy: 1 },
            { class: Call, resource: "Uncalibrated", latency: 3, occupancy: 3 },
            { class: Constant32, resource: "Alu", latency: 2, occupancy: 2 },
            { class: Constant64, resource: "Alu", latency: 8, occupancy: 8 },
            { class: Copy, resource: "Uncalibrated", latency: 1, occupancy: 1 },
            { class: FloatAdd, resource: "Uncalibrated", latency: 4, occupancy: 1 },
            { class: FloatCompare, resource: "Uncalibrated", latency: 3, occupancy: 1 },
            { class: FloatComparePair, resource: "Uncalibrated", latency: 4, occupancy: 2 },
            { class: FloatConvert, resource: "Uncalibrated", latency: 4, occupancy: 1 },
            { class: FloatDiv, resource: "Uncalibrated", latency: 32, occupancy: 32 },
            { class: FloatMul, resource: "Uncalibrated", latency: 5, occupancy: 1 },
            { class: FloatSqrt, resource: "Uncalibrated", latency: 32, occupancy: 32 },
            { class: FloatToInt, resource: "Uncalibrated", latency: 4, occupancy: 1 },
            { class: IntPair, resource: "Alu", latency: 2, occupancy: 2 },
            { class: IntToFloat, resource: "Uncalibrated", latency: 4, occupancy: 1 },
            { class: Load, resource: "LoadStore", latency: 4, occupancy: 1 },
            { class: Return, resource: "Uncalibrated", latency: 1, occupancy: 1 },
            { class: Select, resource: "Uncalibrated", latency: 4, occupancy: 4 },
            { class: Store, resource: "LoadStore", latency: 1, occupancy: 1 },
            { class: Trap, resource: "Uncalibrated", latency: 1, occupancy: 1 },
            { class: IntAlu, resource: "Alu", latency: 1, occupancy: 1 },
            { class: IntMul32, resource: "Mul", latency: 3, occupancy: 1 },
            { class: IntMul64, resource: "Mul", latency: 4, occupancy: 2 },
            // Division is operand-dependent; these are coarse estimates.
            { class: IntDiv32, resource: "Div", latency: 24, occupancy: 24 },
            { class: IntDiv64, resource: "Div", latency: 24, occupancy: 24 },
        ],
    };
}
