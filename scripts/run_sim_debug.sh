DataDir='/project/def-mdiamond/tomren/mathusla/data/fit_study/'
Script='muon_gun_tom_range.mac'
simulation='/project/def-mdiamond/tomren/mathusla/Mu-Simulation/simulation '
${simulation} -o ${DataDir}/eight_layer_test -q -s $Script energy 1000 count 400
