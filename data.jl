using Knet,AutoGrad,ArgParse,Compat,Base

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--path"; nargs='+'; help="If provided, use first file for training, second for dev, others for test.")
    end
    return parse_args(s;as_symbols = true)        
end

function main(args=ARGS)
    opts = parse_commandline()
    println("opts=",[(k,v) for (k,v) in opts]...)
    path = opts[:path]
    directory = path[1]
    batchSize = 64
    contextSize = 20
    filterHeight = 5
    vocabSize = 2000 # change it to 2000 when using complete dataset
    
    # Reading and preparing data
    words = readWords(directory, contextSize, filterHeight)
    wordCounts = createVocabulary(words)
    data, wordToIndex, indexToWord = indexing(wordCounts, vocabSize-1)
    # println(wordToIndex["<s>"])
    x_batches, y_batches = createBatches(data, batchSize, contextSize)

    # Adding Model GCNN
    
    info("$(length(words)) unique chars.")
end

function getFirstNElem(vocab, nElements)
    return sort(collect(vocab), by = tuple -> last(tuple), rev=true)[1:nElements]
end

function printVocab(vocab)
    for (token, count) in vocab
        println(token, " ==> ", count)
    end
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

# Count for each word in all the sentences
function createVocabulary(tokens)
    vocab = Dict{AbstractString,Int}()
    for token in tokens
        vocab[token] = get(vocab, token, 0) + 1
    end
    return vocab
end

# Creating specialized vocabulary for Google Billion Words. Returns sentences(words) by appending "<s>", "</s>" and "unknwon" tokens
function readWords(directory, contextSize, filterHeight)
    words = Any[]
    start = ["<s>"]
    ending = ["</s>"]
    unknown = ["<unk>"]
    pad = "<pad>"
    files = readdir(directory)
    for file in files
        f = open(string(directory,"/",file))
        lines = readlines(f)
        for line in lines
            tokens = split(line)
            if length(tokens) == contextSize-2
                padding = pad ^ trunc(Int64, (filterHeight/2))
                padNumber = Any[]
                push!(padNumber, padding)
                prepend!(tokens, start)
                prepend!(tokens, padNumber)
                append!(tokens, ending) 
                append!(words, tokens)
            end
        end
        close(f)
    end
    return words
end

# Counting word occurences. Returns Dictionary of Vocabulary i.e. tokens and their counts
function wordOccurences(directory, contextSize, filterHeight)
    words = readWords(directory, contextSize, filterHeight)
    vocab = createVocabulary(words)
    for (token,count) in vocab
        if token != "<s>" && token != "</s>" && token != "<unk>" && token != "<pad>" && vocab[token] < contextSize
            get(vocab, "<unk>", 0) + 1
            delete!(vocab, token)
        end
    end
    println(vocab)
    return vocab
end

# Indexing Words
function indexing(words, nElements)
    selectedWords = getFirstNElem(words, nElements)
    wordToIndex = Dict{AbstractString,Int64}()
    indexToWord = Dict{Int64,AbstractString}()
    wordToIndex["<unk>"] = 0
    indexToWord[0] = "<unk>"
    counter = 1
    println(typeof(selectedWords))
    for i in enumerate(selectedWords)
        # println(selectedWords[counter][1])
        wordToIndex[selectedWords[counter][1]] = counter+1
        indexToWord[counter+1] = selectedWords[counter][1]
        counter = counter + 1
    end
    data = []
    for word in keys(words)
        idx = get(wordToIndex, word, -1)
        if (idx == -1)
            idx = wordToIndex["<unk>"]
        else 
            idx = idx
        end
        push!(data, idx)
    end
    return data, wordToIndex, indexToWord
end

# Creating Batches
function createBatches(data, batchSize, contextSize)
    x_batches = Any[]
    y_batches = Any[]

    numBatches = length(data) / (batchSize * contextSize)
    numBatches = trunc(Int64, numBatches)
    data = data[1:(numBatches * batchSize * contextSize)]
    xdata = copy(data)
    # println(length(data))
    # println(batchSize*contextSize)
    ydata = deepcopy(data)

    ydata[end] = xdata[1]
    ydata[1] = xdata[end]
    
    for i in 1:numBatches
        x_batch = xdata[i:batchSize:end]
        y_batch = ydata[i:batchSize:end]
        push!(x_batches, x_batch)
        push!(y_batches, y_batch)
    end
    # println(x_batches[1])
    # println(x_batches[2])
    
    for i in 1:numBatches
        x_batches[i] = x_batches[i][1:end-1]
        y_batches[i] = y_batches[i][1:end-1]
    end
    return x_batches, y_batches
end

function getBatches(x_batches, y_batches, batchID)
    x = x_batches[batchID]
    y = y_batches[batchID]
    batchID = batchID + 1
    if batchID > length(x_batches)
        batchID = 0
    end
    return x, reshape(y,1,size(x)), batchID
end

main()