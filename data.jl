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
    contextSize = 22 + trunc(Int64, filterHeight/2) # Length of sentence/context 20=sentence + 2=<s></s> + trunc(Int64, filterHeight/2)
    vocabSize = 2000                                # change it to 2000 when using complete dataset
    embeddingSize = 200                             # embedding size of each token
    filterWidth = embeddingSize                     # Width of the filter
    
    # Reading and preparing data
    words = readWords(directory, contextSize, filterHeight)
    wordCounts = createVocabulary(words)
    data, wordToIndex, indexToWord, sortedIdToWord = indexing(words, wordCounts, vocabSize-1)
    x_batches, y_batches = createBatches(data, batchSize, contextSize)

    # Embedding Matrix initialization
    embeddingMatrix = createEmbeddings(sortedIdToWord, embeddingSize)

    # Adding Model GCNN
    
    info("$(length(words)) unique chars.")
end

# Creating specialized vocabulary for Google Billion Words. Returns sentences(words) by appending "<s>", "</s>" and "unknwon" tokens
function readWords(directory, contextSize, filterHeight)
    words = Any[]
    start = ["<s>"]
    ending = ["</s>"]
    pad = "<pad>"
    files = readdir(directory)
    for file in files
        f = open(string(directory,"/",file))
        lines = readlines(f)
        for line in lines
            tokens = split(line)
            if length(tokens) == contextSize-2
                padding = trunc(Int64, (filterHeight/2))
                padNumber = Any[]
                for p in 1:padding
                    push!(padNumber, pad)
                end
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

# Count for each word in all the sentences
function createVocabulary(tokens)
    vocab = Dict{AbstractString,Int}()
    for token in tokens
        vocab[token] = get(vocab, token, 0) + 1
    end
    return vocab
end

# Indexing Words
function indexing(totalWords, words, nElements)
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
        sortedIndexToWord[key] = indexToWord[key]
    end
    
    # Creating the input text into ID form. The sequence is in the form of 
    # original texts. However, the words are replaced with IDx. If a word does 
    # not match an ID, (i.e. if a word is not in the threshold of nElements [in selectedWords])
    # then that word is given the ID 1, i.e. the word is known as "<unk>".
    # println(length(totalWords))
    for word in totalWords
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

# Create Embeddings
function createEmbeddings(indexToWord, embeddingSize)
    embeddingMatrix = Any[]
    for wordId in keys(indexToWord)
        embedding = randn(1, embeddingSize)
        push!(embeddingMatrix, embedding)
    end
    return embeddingMatrix
end

# Create Weights and biasses
# h(X) = (X * W + b) .* sigm(X * V + c)
# W = w[1], b = w[2], V = w[3], c = w[4]
function initWeights(k, embeddingSize, winit, vocabSize)
    w = Any[(randn(Float32, k,1,embeddingSize,embeddingSize)*winit), zeros(Float32, 1,1,embeddingSize,1)
            (randn(Float32, k,1,embeddingSize,embeddingSize)*winit), zeros(Float32, 1,1,embeddingSize,1)
            (randn(Float32, embeddingSize,vocabSize)*winit), zeros(Float32, 1,vocabSize)]
    return w
end

# Hidden Layers
function hiddenLayers(weights, input)
    conv_w = conv4(weights[1], inputs, padding=2)
    conv_v = conv4(weights[3], inputs, padding=2)
    out = (conv_w .+ weights[2]) .* sigm(conv_v .+ weights[4])
    return out
end

# Hidden Layers
function predict(weights, input, numLayers)
    out = input
    for i in 1:numLayers
        out = hiddenLayers(weights, out)
    end
    
    # TODO: fully connected
    


    return out
end

# Loss function (softmax)
# Should be adaptive softmax according to the paper
function loss(weights, input, ygold, numLayers)
    ypred = predict(weights, input, numLayers)
    ynorm = logp(ypred, 1)
    y = -sum(ygold .* ynorm)/size(ygold, 1)
    return y
end

lossgradient = grad(loss)

function train()
    for (x,y) in dtrn
        loss_grad = lossgradient(w,convert(KnetArray{Float32},x),convert(KnetArray{Float32},y))
        x = convert(KnetArray{Float32}, x)
        opts = map(x->Adam(), w)
        update!(w, loss_grad, opts)
        for k in 1:length(w)
            w[k] = w[k] - (lr * loss_grad[k])
        end 
    end
end

# Utility functions
function printVocab(vocab)
    for (token, count) in vocab
        println(token, " ==> ", count)
    end
end

function getFirstNElem(vocab, nElements)
    return sort(collect(vocab), by = tuple -> last(tuple), rev=true)[1:nElements]
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

# Creating Batches
function createBatches(data, batchSize, contextSize)
    x_batches = Any[]
    y_batches = Any[]

    numBatches = length(data) / (batchSize * contextSize)
    numBatches = trunc(Int64, numBatches)
    data = data[1:(numBatches * batchSize * contextSize)]
    
    xdata = copy(data)
    ydata = deepcopy(data)

    ydata[end] = xdata[1]
    ydata[1] = xdata[end]
    
    start = 0
    iter = 1
    for i in 1:numBatches
        x_batch = Any[]
        y_batch = Any[]
        for j in 1:batchSize
            x_sentence = xdata[start+1:start+contextSize]
            y_sentence = ydata[start+1:start+contextSize]
            push!(x_batch, x_sentence)
            push!(y_batch, y_sentence)
            start = iter*contextSize
            iter += 1
        end
        push!(x_batches, x_batch)
        push!(y_batches, y_batch)
    end

    for i in 1:numBatches
        for j in 1:batchSize
            x_batches[i][j] = x_batches[i][j][1:end-1]
            y_batches[i][j] = y_batches[i][j][1:end-1]
        end
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

main()