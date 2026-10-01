import "abi.spec";
import "features.spec";
import "lower.spec";
// Even the generic model explicitly owns its estimates and execution resources.
cpu generic {
    name = "generic";
    features = [I, M, F, D];
    schedule = {
        issue_width: 1,
        resources: [{ name: "Scalar", units: 1 }],
        classes: [
            { name: "IntAlu", resource: "Scalar", latency: 1, occupancy: 1 },
            { name: "IntMul32", resource: "Scalar", latency: 3, occupancy: 1 },
            { name: "IntMul64", resource: "Scalar", latency: 4, occupancy: 1 },
            { name: "IntDiv32", resource: "Scalar", latency: 20, occupancy: 1 },
            { name: "IntDiv64", resource: "Scalar", latency: 20, occupancy: 1 },
        ],
    };
}
cpu c908 {
    name = "c908";
    features = [I, M, F, D, Zba, Zbb];
    // Coarse integer model, calibrated on K230.
    schedule = {
        issue_width: 2,
        resources: [
            { name: "Alu", units: 2 },
            { name: "Mul", units: 1 },
            { name: "Div", units: 1 },
        ],
        classes: [
            { name: "IntAlu", resource: "Alu", latency: 1, occupancy: 1 },
            { name: "IntMul32", resource: "Mul", latency: 3, occupancy: 1 },
            { name: "IntMul64", resource: "Mul", latency: 4, occupancy: 2 },
            // Division is operand-dependent; these are coarse estimates.
            { name: "IntDiv32", resource: "Div", latency: 24, occupancy: 24 },
            { name: "IntDiv64", resource: "Div", latency: 24, occupancy: 24 },
        ],
    };
}
