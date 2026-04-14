#!/usr/bin/env -S julia --color=yes --startup-file=no
"""
Transfer matrix code module.

Module runs and returns transfer matricies. These are computed if needed.

"""

module tmm_ppv

using CSV,
    DataFrames,
    Interpolations,
    JSON,
    LaTeXStrings,
    Plots,
    Serialization,
    Statistics,
    ThinFilmsTools

gr(size = (171, 120))

global DEFAULT_OUTPUT_TMM_FILENAME::AbstractString = "tmm.csv"
global NK_DATABASE_FILENAME::AbstractString = "winch_mixed_nk_database.csv"

struct Layer
    material::Symbol
    thickness::Number
end

# # Adjust sqrt
# import Base
# import Core.Intrinsics: sqrt_llvm

# Base.sqrt(x::Float64) = x < zero(x) ? Base.sqrt(Complex(x)) : sqrt_llvm(x)
# Base.sqrt(x::Float32) = x < zero(x) ? Base.sqrt(Complex(x)) : sqrt_llvm(x)

stack_name = "stack"
nk_filename = NK_DATABASE_FILENAME
θ::Vector{Real} = [0.0]


"""
    load_stack(stack_name::String)::Vector{Layer}

Load a stack from an input file provided.

### Inputs:
- `stack_name`  -- The name of the stack file to parse

### Outputs:
The stack.

"""
function load_stack(stack_name::String)::Vector{Layer}
    # If the stack file is not a file, throw an error.
    stack_file = stack_name * ".json"
    if !isfile(stack_file)
        throw("Input file not found: " * stack_name * ".json")
    end
    # Otherwise, load the layer information from the file.
    return JSON.parsefile(stack_file, Vector{Layer})
end


"""
    tmm(
        stack::Vector{Layer},
        nk_database::DataFrame,
        θ::Vector{<:Real},
        λmin::Real,
        λmax::Real,
    )

Compute the TMM for a given stack name file, angles, and wavelength limits. This TMM is
then saved and is returned.

**Note:** This function is the base function which is called via various implementations
which each account for a reduced set of input parameters.

### Inputs
- `stack`       -- The stack to use.
- `nk_database` -- The n-k database to use.
- `θ`           -- The angles of incidence for the incoming light.
- `λmin`        -- The initial wavelength to use for the range of wavelengths.
- `λmax`        -- The final wavelength to use for the range of wavelengths.

### Outputs
The transfer matrix.

"""
function tmm(
    stack::Vector{Layer},
    nk_database::DataFrame,
    θ::Vector{<:Real},
    λmin::Real,
    λmax::Real,
)
    # Construct a wavelength series.
    λ::Vector{Real} = (λmin+1):(λmax-1)
    beam = PlaneWave(λ, θ)

    # Construct layers for the TMM.
    layers::Vector{LayerTMMO} = [LayerTMMO(RIdb.air(beam.λ))]

    # Construct an interpolated database and add layers to the TMM
    nk_database_interpolated = DataFrame(λ = collect(λ))
    for layer in stack
        layer_name = String(layer.material)
        linint_n = LinearInterpolation(
            collect(skipmissing(nk_database[:, layer_name*"_wavelength"])),
            collect(skipmissing(nk_database[:, layer_name*"_n"])),
        )
        nk_database_interpolated[!, layer_name*"_n"] = linint_n(λ)
        linint_k = LinearInterpolation(
            collect(skipmissing(nk_database[:, layer_name*"_wavelength"])),
            collect(skipmissing(nk_database[:, layer_name*"_k"])),
        )
        nk_database_interpolated[!, layer_name*"_k"] = linint_k(λ)
        push!(
            layers,
            LayerTMMO(
                nk_database_interpolated[!, layer_name*"_n"] +
                (nk_database_interpolated[!, layer_name*"_k"] .* im),
                type = :GT,
                d = Float64(layer.thickness),
            ),
        )
    end

    push!(layers, LayerTMMO(RIdb.air(beam.λ)))

    # Compute the TMM.
    sol = tmm_optics(beam, layers; λ0 = mean(beam.λ), emfflag = true, h = 10);

    # figure = plot(
    #     Spectrum1D(),
    #     sol.Beam.λ,
    #     [sol.Spectra.Rp, sol.Spectra.Tp, 1.0 .- (sol.Spectra.Rp .+ sol.Spectra.Tp)],
    #     label = ["Reflectance" "Transmittance" "Absorbance"],
    #     line = ([:solid :dash :dashdot]),
    #     ylims = (0.0, 1.0),
    #     xlims = (sol.Beam.λ[1], sol.Beam.λ[end]),
    #     palette = palette(:devon, 5),
    #     size=(171, 120),
    # );
    # xlabel!("Wavelength / nm")
    # ylabel!("Fraction refelcted, transmitted or absorbed")
    # savefig(figure, "tmm.pdf")

    # figure = plot(
    #     EMF2D(),
    #     sol.Beam.λ,
    #     sol.Misc.ℓ,
    #     log10.(sol.Field.emfp[:, 1, :]),
    #     title = ("Log of EMF intensity"),
    #     size=(171,120),
    #     palette=:viridis,
    # );
    # xlabel!("Wavelength / nm")
    # ylabel!("Fraction refelcted, transmitted or absorbed")
    # savefig(figure, "emf.pdf")
    # gui()

    return sol

end


"""
    tmm(stack_name::AbstractString, θ::Vector{<:Real}, nk_filename::AbstractString)

Compute the TMM for a given stack name file and angle. The wavelength ranges utilise
default assumed angles. This TMM is then saved and is returned.

### Inputs
- `stack_name`  -- The name of the file to use for loading the stack information from.
- `θ`           -- The angle of incidence for the incoming light.
- `nk_filename` -- The filename for the n-k database.

### Outputs
The transfer matrix.

"""
function tmm(stack_name::AbstractString, θ::Vector{<:Real}, nk_filename::AbstractString)
    # Parse the stack.
    stack = load_stack(stack_name)

    # Parse the n-k data for the stack.
    nk_database = CSV.read(nk_filename, DataFrame)

    # Confirm that all of the layers exist in the database.
    database_columns = names(nk_database)
    for layer in stack
        layer_name = String(layer.material)
        if layer_name * "_wavelength" ∉ database_columns ||
           layer_name * "_n" ∉ database_columns ||
           layer_name * "_k" ∉ database_columns
            throw("Layer '" * layer_name * "' is missing data in database.")
        end
    end

    # Determine the minimum spanning wavelength range for the stack.
    nk_description = describe(
        nk_database[
            !,
            names(nk_database)[[
                any(occursin.("wavelength", column)) for column ∈ database_columns
            ]],
        ],
    )
    λmin = Int(trunc(maximum(nk_description.min)))
    λmax = Int(ceil(minimum(nk_description.max)))

    return tmm(stack, nk_database, θ, λmin, λmax)
end

"""
    tmm(stack_name::AbstractString, θ::Vector{<:Real})

Compute the TMM for a given stack name file and array of angles. The wavelength ranges
utilise default assumed angles. This TMM is then saved and is returned.

### Inputs
- `stack_name`  -- The name of the file to use for loading the stack information from.
- `θ`           -- The array of angles of incidence for the incoming light.

### Outputs
The transfer matrix.

"""
function tmm(stack_name::String, θ::Vector{<:Real})
    return tmm(stack_name, θ, NK_DATABASE_FILENAME)
end

"""
    tmm(stack_name::AbstractString, θ::Real)

Compute the TMM for a given stack name file and angle. The wavelength ranges utilise
default assumed angles. This TMM is then saved and is returned.

### Inputs
- `stack_name`  -- The name of the file to use for loading the stack information from.
- `θ`           -- The angle of incidence for the incoming light.

### Outputs
The transfer matrix.

"""
function tmm(stack_name::String, θ::Real)
    return tmm(stack_name, [θ])
end

"""
    tmm(stack_name::AbstractString, θ::Number)

Compute the TMM for a given stack name file and angle. The wavelength ranges utilise
default assumed angles. This TMM is then saved and is returned.

### Inputs
- `stack_name`  -- The name of the file to use for loading the stack information from.
- `θ`           -- The angle of incidence for the incoming light.

### Outputs
The transfer matrix.

"""
function tmm(stack_name::String, θ::UnitRange)
    return tmm(stack_name, Vector(θ))
end

"""
    tmm(stack_name::AbstractString)

Compute the TMM for a given stack name file. The wavelength ranges utilise default assumed
angles whilst the angle of incidence is assumed to be normal. This TMM is then saved and is
returned.

### Inputs
- `stack_name`  -- The name of the file to use for loading the stack information from.

### Outputs
The transfer matrix.

"""
function tmm(stack_name::String)
    return tmm(stack_name, 0)
end

"""
    tmm_to_file(stack_name::String, θ::Vector{<:Real}, output_name::String)

Comuptes the TMM and saves the result to a dataframe file.

### Inputs
- `stack_name`  -- The name of the file to use for loading the stack information from.
- `θ`           -- The angles of incidence to use.
- `output_name` -- The name of the output file to use.

"""
function tmm_to_file(stack_name::String, θ::Vector{<:Real}, output_name::String)
    # Run the TMM model
    tmm_data = tmm(stack_name, θ)

    if !occursin(".csv", output_name)
        output_name *= ".csv"
    end

    # Save output results
    tmm_data_frame = hcat(
        DataFrame(wavelength = tmm_data.Beam.λ),
        DataFrame(tmm_data.Spectra.Tp, [repr(entry) for entry in θ]),
    )
    CSV.write(output_name, tmm_data_frame)

end

"""
    tmm_to_file(stack_name::String, θ::Vector{<:Real})

Comuptes the TMM and saves the result to a dataframe file.

### Inputs
- `stack_name`  -- The name of the file to use for loading the stack information from.
- `θ`           -- The angles of incidence to use.

"""
function tmm_to_file(stack_name::String, θ::Vector{<:Real})
    tmm_to_file(stack_name, θ, DEFAULT_OUTPUT_TMM_FILENAME)

end

"""
    tmm_to_file(stack_name::String, θ::AbstractVectpr{Any}, output_name::String)

Comuptes the TMM and saves the result to a dataframe file.

### Inputs
- `stack_name`  -- The name of the file to use for loading the stack information from.
- `θ`           -- The angles of incidence to use.
- `output_name` -- The name of the output file to use.

"""
function tmm_to_file(stack_name::String, θ::AbstractVector{Any}, output_name::String)
    tmm_to_file(stack_name, Vector{<:Real}(θ), output_name)

end

"""
    tmm_to_file(stack_name::String, θ::UnitRange, output_name::String)

Comuptes the TMM and saves the result to a dataframe file.

### Inputs
- `stack_name`  -- The name of the file to use for loading the stack information from.
- `θ`           -- The angles of incidence to use.
- `output_name` -- The name of the output file to use.
"""
function tmm_to_file(stack_name::String, θ::UnitRange, output_name::String)
    tmm_to_file(stack_name, Vector(θ), output_name)

end

"""
    tmm_to_file(stack_name::String, θ::Real)

Comuptes the TMM and saves the result to a dataframe file.

### Inputs
- `stack_name`  -- The name of the file to use for loading the stack information from.
- `θ`           -- The angle of incidence to use.
- `output_name` -- The name of the output file to use.
"""
function tmm_to_file(stack_name::String, θ::Real, output_name::String)
    tmm_to_file(stack_name, [θ], DEFAULT_OUTPUT_TMM_FILENAME)

end

"""
    tmm_to_file(stack_name::String, θ::UnitRange)

Comuptes the TMM and saves the result to a dataframe file.

### Inputs
- `stack_name`  -- The name of the file to use for loading the stack information from.
- `θ`           -- The angles of incidence to use.
"""
function tmm_to_file(stack_name::String, θ::UnitRange)
    tmm_to_file(stack_name, Vector(θ), DEFAULT_OUTPUT_TMM_FILENAME)

end

"""
    tmm_to_file(stack_name::String, θ::Real)

Comuptes the TMM and saves the result to a dataframe file.

### Inputs
- `stack_name`  -- The name of the file to use for loading the stack information from.
- `θ`           -- The angle of incidence to use.
"""
function tmm_to_file(stack_name::String, θ::Real)
    tmm_to_file(stack_name, [θ], DEFAULT_OUTPUT_TMM_FILENAME)

end

"""
    tmm_to_file(stack_name::String, output_name::String)

Comuptes the TMM and saves the result to a dataframe file.

### Inputs
- `stack_name`  -- The name of the file to use for loading the stack information from.
- `output_name` -- The name of the output file to use.
"""
function tmm_to_file(stack_name::String, output_name::String)
    tmm_to_file(stack_name, [0], output_name)

end

"""
    tmm_to_file(stack_name::String)

Comuptes the TMM and saves the result to a dataframe file.

### Inputs
- `stack_name`  -- The name of the file to use for loading the stack information from.
"""
function tmm_to_file(stack_name::String)
    tmm_to_file(stack_name, DEFAULT_OUTPUT_TMM_FILENAME)

end

end
