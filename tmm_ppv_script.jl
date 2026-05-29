
#!/usr/bin/env -S julia --color=yes --startup-file=no
"""
Transfer matrix code script.

Script runs and returns transfer matricies. These are computed if needed.

"""

using ArgParse

include("tmm_ppv.jl")

# Enable parsing of unit range
function ArgParse.parse_item(::Type{Union{Vector{<:Number},Number}}, x::AbstractString)
    # If a colon is present, then try to parse as a unit range.
    if occursin(":", x)
        bounds = split(x, ":")
        if length(bounds) > 3
            throw(
                BoundsError(
                    "Unit range must have one or two colons. Bounds: " *
                    string(bounds) *
                    "; Len(bounds): " *
                    string(length(bounds)),
                ),
            )
        end
        if length(bounds) == 3
            return Vector(parse(Int, bounds[1]):parse(Int, bounds[2]):parse(Int, bounds[3]))
        end
        return Vector(parse(Int, bounds[1]):parse(Int, bounds[2]))
    end

    # If a comma is present, try to parse as a vector.
    if occursin(",", x)
        _int_regex = r"([0-9])"
        return [parse(Int64, t.match) for t in eachmatch(_int_regex, x)]
    end

    # Else, parse as a number.
    return parse(Float64, x)
end


"""
    parse_commandline()

Parses the CLI into a series of parsed arguments.

### Outputs
The parsed arguments.
"""
function parse_commandline()
    argparse_settings = ArgParseSettings()

    # Add the argument table.
    @add_arg_table! argparse_settings begin
        "--stack-name", "-s"
        help = "The name of the stack to use."
        arg_type = String
        default = "stack"
        "--to-file", "-f"
        help = "Run the TMM and save to a file."
        arg_type = String
        "--theta", "-t"
        help = "The angle(s) to use."
        arg_type = Union{Vector{<:Number},Number}
        default = 0
        "--wavelength-resolution", "-w"
        help = "The unit to use for the wavealength."
        arg_type = Float64
        default = 1
        # "--flag1"
        #     help = "an option without argument, i.e. a flag"
        #     action = :store_true
        # "arg1"
        #     help = "a positional argument"
        #     required = true
    end

    return parse_args(argparse_settings)
end

"""
    main()

Main function for when being run as a script.
"""
function main()
    # Parse the CLI arguments.
    parsed_args = parse_commandline()

    # Confirm the arguments to the user.
    println("Launching Julia TMM calculation --- Parsed args:")
    for (arg, val) in parsed_args
        println("  $arg  =>  $val")
    end

    # If the to-file arg is specified, run the to-file script.
    if !isnothing(parsed_args["to-file"])
        try
            tmm_ppv.tmm_to_file(
                parsed_args["stack-name"],
                parsed_args["theta"],
                parsed_args["to-file"],
                parsed_args["wavelength-resolution"],
            )
        catch err
            println("TMM script failed.")
            throw(err)
        else
            println("TMM script completed and successfully saved to file.")
        end
    end
end

main()
