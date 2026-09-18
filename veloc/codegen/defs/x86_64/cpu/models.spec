import "features.spec";

cpu generic { name = "generic"; features = []; }
cpu haswell { name = "haswell"; features = [SSE41, AVX, AVX2, BMI1, BMI2, POPCNT]; }
cpu skylake { name = "skylake"; features = [SSE41, AVX, AVX2, BMI1, BMI2, POPCNT]; }
