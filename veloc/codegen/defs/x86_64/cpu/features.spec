import "../predicates.spec";

feature SSE41 { doc = "Intel SSE4.1 Instructions"; requires = []; }
feature AVX { doc = "Intel Advanced Vector Extensions"; requires = []; }
feature AVX2 { doc = "Intel Advanced Vector Extensions 2"; requires = [AVX]; }
feature BMI1 { doc = "Bit Manipulation Instruction Set 1"; requires = []; }
feature BMI2 { doc = "Bit Manipulation Instruction Set 2"; requires = []; }
extractor HAS_BMI2 { params = []; pattern = has_bmi2(); }
extractor HAS_AVX2 { params = []; pattern = has_avx2(); }
feature POPCNT { doc = "Population count instruction"; requires = []; }
