using Statistics

function error_summary(x_err, unit)
    mean = Statistics.mean(x_err)
    std_dev = Statistics.std(x_err)
    return "Mean: $(mean) $unit\nStd. Dev.: $(std_dev) $unit"
end