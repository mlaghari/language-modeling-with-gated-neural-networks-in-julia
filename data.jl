using Knet,AutoGrad,ArgParse,Compat

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--datafiles"; nargs='+'; help="If provided, use first file for training, second for dev, others for test.")
        # ("--generate"; arg_type=Int; default=500; help="If non-zero generate given number of characters.")
        # ("--hidden";  arg_type=Int; default=256; help="Sizes of one or more LSTM layers.")
        # ("--epochs"; arg_type=Int; default=3; help="Number of epochs for training.")
        # ("--batchsize"; arg_type=Int; default=100; help="Number of sequences to train on in parallel.")
        # ("--seqlength"; arg_type=Int; default=25; help="Number of steps to unroll the network for.")
        # ("--decay"; arg_type=Float64; default=0.9; help="Learning rate decay.")
        # ("--lr"; arg_type=Float64; default=1e-1; help="Initial learning rate.")
        # ("--gclip"; arg_type=Float64; default=3.0; help="Value to clip the gradient norm at.")
        # ("--winit"; arg_type=Float64; default=0.1; help="Initial weights set to winit*randn().")
        # ("--gcheck"; arg_type=Int; default=0; help="Check N random gradients.")
        # ("--seed"; arg_type=Int; default=38; help="Random number seed.")
        # ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
    end
    return parse_args(s;as_symbols = true)        
end

function main(args=ARGS)
    opts = parse_commandline()
    println("opts=",[(k,v) for (k,v) in opts]...)
    vocab = wordOccurences(path[1], 3)

    info("$(length(vocab)) unique chars.") # The output should be 65 for input.txt 
end


function createVocabulary(tokens)
    vocab = Dict{AbstractString,Int}()
    for token in tokens
        vocab[token] = get(vocab, token, 0) + 1
    end
    return vocab
end

# Cummulative tokens of all files in a directory
function createTokens(directory)
    tokens = []
    files = readdir(directory)
    for file in files
        f = open(file)
        content = readstring(f)
        push!(tokens, split(content))
        close(f)
    end
    return tokens
end

# Creating specialized vocabulary for Google Billion Words
function tokensForGBW(directory, contextSize)
    words = Any[]
    start = ["<s>"]
    ending = ["</s>"]
    unknown = ["<unknown>"]
    files = readdir(directory)
    # println(files)
    for file in files
        f = open(string(directory,"/",file))
        lines = readlines(f)
        for line in lines
            tokens = split(line)
            # if len(tokens) == contextSize-2
                prepend!(tokens, start)
                append!(tokens, ending)
                append!(tokens, unknown)
                append!(words, tokens)
            # end
        end
        close(f)
    end
    return words
end

# Counting word occurences
function wordOccurences(directory, limit)
    words = tokensForGBW(directory, -1)
    vocab = createVocabulary(words)
    for (token,count) in vocab
        if token != "<s>" && token != "</s>" && token != "<unknown>" && vocab[token] < limit
            get(vocab, "<unknown>", 0) + 1
            delete!(vocab, token)
        end
    end
    return vocab
end

main()