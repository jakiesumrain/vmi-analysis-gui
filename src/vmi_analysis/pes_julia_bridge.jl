# Adapter to the public PBASEX estimators. The Python launcher validates inputs
# and publishes the CSV only after this process returns successfully.
using PBASEX
using PBASEX.ImageProcessing: rotated_grid
using Random: Xoshiro

function csv_row(io, kind, quantity, low, high, coordinate, order, value, se;
                 ci=("", ""), valid="", invalid="")
    println(io, join((kind, quantity, low, high, coordinate, order, value, se, ci..., valid, invalid), ','))
end

function write_windows(io, windows, orders; bootstrap=false)
    for w in windows
        low, high = w.window
        width = high - low
        center = low + width / 2
        diagnostics = bootstrap ? (; valid=w.valid_replicates, invalid=w.invalid_fraction) : (;)
        yield_ci = bootstrap ? w.yield_ci : ("", "")
        spectrum_ci = bootstrap ? yield_ci ./ width : ("", "")
        csv_row(io, "window", "yield", low, high, center, 0, w.yield, w.yield_se;
                ci=yield_ci, diagnostics...)
        csv_row(io, "window", "spectrum", low, high, center, 0, w.spectrum, w.spectrum_se;
                ci=spectrum_ci, diagnostics...)
        for (j, order) in enumerate(orders)
            ci = bootstrap ? w.beta_ci[j] : ("", "")
            csv_row(io, "window", "beta", low, high, center, order, w.betas[j], w.beta_se[j];
                    ci, diagnostics...)
        end
    end
end

function write_native(io, r, orders; spectrum_se=nothing, beta_se=nothing)
    for i in eachindex(r.E)
        csv_row(io, "native", "spectrum", "", "", r.E[i], 0, r.IE[i],
                isnothing(spectrum_se) ? "" : spectrum_se[i])
        for (j, order) in enumerate(orders)
            csv_row(io, "native", "beta", "", "", r.E[i], order, r.betas[i,j],
                    isnothing(beta_se) ? "" : beta_se[i,j])
        end
    end
end

function main(args)
    mode, source, basis, output, alpha, reg, rotation, x, y = args[1:9]
    center = isempty(x) ? nothing : (parse(Int, x), parse(Int, y))
    settings = (; basis, center, alpha=parse(Float64, alpha), reg=parse(Float64, reg),
                 rotation=parse(Int, rotation))
    bootstrap = mode == "shots"
    kind = args[15]
    windows = [Tuple(parse.(Float64, split(line))) for line in eachline(args[14])]
    analysis = if bootstrap
        hasmethod(pbasex_shots, Tuple{typeof(source)}, (:project_shots,)) ||
            throw(ArgumentError("Shot bootstrap requires a PBASEX-Julia checkout that supports project_shots."))
        window_batch = parse(Int, args[16])
        batch_count = cld(length(windows), window_batch)
        batch_label = isone(batch_count) ? "batch" : "batches"
        println("Processing $(length(windows)) energy windows in $batch_count $batch_label with per-shot projection (up to $window_batch per batch).")
        flush(stdout)
        pbasex_shots(source; settings..., windows, replicates=parse(Int, args[10]),
                     confidence=parse(Float64, args[11]), block_length=parse(Int, args[12]),
                     rng=Xoshiro(parse(Int, args[13])), window_batch, project_shots=true)
    else
        resolution = parse(Float64, args[10])
        counts = read!(source, Matrix{Float64}(undef, parse(Int, args[11]), parse(Int, args[12])))
        geometry = Symbol(args[13])
        scale = settings.alpha / resolution^2
        isfinite(scale) && scale > 0 || throw(ArgumentError("Invalid energy scale after count-image resolution"))
        if kind == "spectrum"
            rows, cols = size(counts)
            origin = something(center, (cols ÷ 2 + 1, rows ÷ 2 + 1))
            turns = mod(settings.rotation ÷ 90, 4)
            grid = rotated_grid(size(counts), origin, turns)
            pbasex(iszero(turns) ? counts : rotl90(counts, turns);
                   basis, geometry, center=grid.center, alpha=scale, reg=settings.reg, make_images=false)
        elseif kind == "peaks"
            pbasex_counts(counts; settings..., alpha=scale, geometry, windows, make_images=false, native_errors=false)
        else
            pbasex_counts(counts; settings..., alpha=scale, geometry, windows, make_images=false)
        end
    end
    bootstrap && println("Bootstrapped $(analysis.shots) shots with $(analysis.bootstrap_replicates) replicates.")
    orders = analysis.legendre_orders[2:end]
    open(output, "w") do io
        println(io, "kind,quantity,lower,upper,coordinate,order,value,se,ci_lower,ci_upper,valid_replicates,invalid_fraction")
        if kind == "spectrum"
            write_native(io, analysis, orders)
        else
            if !bootstrap && kind == "legacy"
                write_native(io, analysis.result, orders; analysis.spectrum_se, analysis.beta_se)
            end
            write_windows(io, analysis.windows, orders; bootstrap)
        end
    end
end

main(ARGS)
