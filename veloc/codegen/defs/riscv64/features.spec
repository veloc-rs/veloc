feature I { doc = "RV64 base integer instructions"; requires = []; }
feature C { doc = "Compressed integer and floating-point encodings"; requires = [I]; }
feature M { doc = "Integer multiply and divide"; requires = [I]; }
feature F { doc = "Single-precision floating point"; requires = [I]; }
feature D { doc = "Double-precision floating point"; requires = [F]; }
feature Zba { doc = "Address generation"; requires = [I]; }
feature Zbb { doc = "Basic bit manipulation"; requires = [I]; }
