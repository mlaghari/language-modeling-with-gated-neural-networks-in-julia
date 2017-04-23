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
    batchSize = 64                                  # Batch size of data while training
    filterHeight = 5                                # Height of the CNN filter
    contextSize = 20 + trunc(Int64, filterHeight/2) # Length of sentence/context
    vocabSize = 2000                                # change it to 2000 when using complete dataset
    embeddingSize = 200                             # embedding size of each token
    filterWidth = embeddingSize                     # Width of the filter
    
    # Reading and preparing data
    words = readWords(directory, contextSize, filterHeight)
    wordCounts = createVocabulary(words)
    data, wordToIndex, indexToWord, sortedIdToWord = indexing(wordCounts, vocabSize-1)
    x_batches, y_batches = createBatches(data, batchSize, contextSize)

    # Embedding Matrix initialization
    embeddingMatrix = createEmbeddings(sortedIdToWord, embeddingSize)
    println(embeddingMatrix[1])

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
    # selectedWords is a 2d array, Array{AbstractString,Int64} containing 
    # the words and its counts
    selectedWords = getFirstNElem(words, nElements)
    wordToIndex = Dict{AbstractString,Int64}()
    indexToWord = Dict{Int64,AbstractString}()
    sortedIndexToWord = Dict{Int64,AbstractString}()
    data = []
    
    wordToIndex["<unk>"] = 1
    indexToWord[1] = "<unk>"
    counter = 1
    for i in enumerate(selectedWords)
        wordToIndex[selectedWords[counter][1]] = counter+1
        indexToWord[counter+1] = selectedWords[counter][1]
        counter = counter + 1
    end
    
    for key in sort(collect(keys(indexToWord)))
        # println("$key => $(indexToWord[key])")
        sortedIndexToWord[key] = indexToWord[key]
    end
    
    # Creating the input text into ID form. The sequence is in the form of 
    # original texts. However, the words are replaced with IDx. If a word does 
    # not match an ID, (i.e. if a word is not in the threshold of nElements [in selectedWords])
    # then that word is given the ID 1, i.e. the word is known as "<unk>".
    for word in keys(words)
        idx = get(wordToIndex, word, -1)
        if (idx == -1)
            idx = wordToIndex["<unk>"]
        else 
            idx = idx
        end
        push!(data, idx)
    end

    return data, wordToIndex, indexToWord, sortedIndexToWord
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

# Get a particular batch according to BatchID
function getBatches(x_batches, y_batches, batchID)
    x = x_batches[batchID]
    y = y_batches[batchID]
    batchID = batchID + 1
    if batchID > length(x_batches)
        batchID = 0
    end
    return x, reshape(y,1,size(x)), batchID
end

# Create Embeddings
function createEmbeddings(indexToWord, embeddingSize)
    embeddingMatrix = Any[]
    for wordId in keys(indexToWord)
        embedding = randn(embeddingSize, 1)
        embedding = transpose(embedding)
        push!(embeddingMatrix, embedding)
    end
    return embeddingMatrix
end

# Create Weights and biasses
function initParams(k, m, n)
    # w = Any[(randn())]
end

main()