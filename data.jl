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
    filterWidth = 5                                 # Width of the filter
    numHiddenLayers = 10
    gradientClip = 0.1
    
    # Reading and preparing data
    words = readWords(directory, contextSize, filterHeight)
    wordCounts = createVocabulary(words)
    data, wordToIndex, indexToWord, sortedIdToWord = indexing(words, wordCounts, vocabSize-1)
    x_batches, y_batches = createBatches(data, batchSize, contextSize, indexToWord)
    info("$(length(words)) unique chars.")

    # Embedding Matrix initialization
    embeddingMatrix = createEmbeddings(sortedIdToWord, embeddingSize)

    # Adding Model GCNN
    w = initWeights(filterWidth, embeddingSize, 0.01, vocabSize)
    for i in 1:size(x_batches, 1)
        w = train(w, x_batches[i], y_batches[i], gradientClip, embeddingMatrix, vocabSize, numHiddenLayers)
    end

end

# Creating specialized vocabulary for Google Billion Words. 
# Returns sentences(words) by appending "<s>", "</s>" and "<pad>" tokens
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
            if length(tokens) == contextSize
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
    # Length of each sentence is not 2+1+contextSize+1
    # 2 for padding (can vary)
    # 1 is for <s>
    # 1 is for </s>
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
function indexing(words, wordCounts, nElements)
    # selectedWords is a 2d array, Array{AbstractString,Int64} containing 
    # the words and its counts
    selectedWords = getFirstNElem(wordCounts, nElements)
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
    for word in words
        idx = get(wordToIndex, word, -1)
        if (idx == -1)
            idx = wordToIndex["<unk>"]
        else 
            idx = idx
        end
        push!(data, idx)
    end
    # println("Indices of self-added tokens")
    # println(wordToIndex["<pad>"])
    # println(wordToIndex["<s>"])
    # println(wordToIndex["</s>"])
    # println("----------------------------")
    return data, wordToIndex, indexToWord, sortedIndexToWord
end

# Creating Batches
function createBatches(data, batchSize, contextSize, idToWords)
    x_batches = Any[]
    y_batches = Any[]

    numBatches = length(data) / (batchSize * ((contextSize+4)))
    numBatches = trunc(Int64, numBatches)
    data = data[1:(numBatches * batchSize * ((contextSize+4)))]
    
    xdata = copy(data)
    ydata = deepcopy(data)

    ydata[1:end-1] = xdata[2:end]
    ydata[end] = xdata[1]
    # for i in 1:(contextSize+4)
    #     println(xdata[i], " ==> ", ydata[i])
    # end
    start = 0
    iter = 1
    for i in 1:numBatches
        x_batch = Any[]
        y_batch = Any[]
        for j in 1:batchSize
            x_sentence = xdata[start+1:start+(contextSize+4)]
            y_sentence = ydata[start+1:start+(contextSize+4)]
            push!(x_batch, x_sentence)
            push!(y_batch, y_sentence)
            start = iter*(contextSize+4)
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
    # for i in 1:(contextSize+3)
    #     println(idToWords[x_batches[1][1][i]], " ==> ", idToWords[y_batches[1][1][i]])
    # end
    println(size(x_batches), " ", size(x_batches[1]), " ", size(x_batches[1][1]))
    return x_batches, y_batches
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
    w = Any[KnetArray{Float32}(randn(Float32, k,1,embeddingSize,embeddingSize)*winit), KnetArray{Float32}(zeros(Float32, 1,1,embeddingSize,1)),
            KnetArray{Float32}(randn(Float32, k,1,embeddingSize,embeddingSize)*winit), KnetArray{Float32}(zeros(Float32, 1,1,embeddingSize,1)),
            KnetArray{Float32}(randn(Float32, embeddingSize,vocabSize)*winit), KnetArray{Float32}(zeros(Float32, 1,vocabSize))]
    # return map(t->convert(KnetArray{Float32}, t), w)
    return w
end

# Hidden Layers
function hiddenLayers(weights, inputs)
    conv_w = conv4(convert(KnetArray{Float32}, weights[1]), convert(KnetArray{Float32}, inputs), padding=2)
    conv_v = conv4(convert(KnetArray{Float32}, weights[3]), convert(KnetArray{Float32}, inputs), padding=2)
    out = (conv_w .+ weights[2]) .* sigm(conv_v .+ weights[4])
    return out
end

# Hidden Layers
function predict(weights, input, numLayers)
    out = input
    for i in 1:numLayers-1
        out = hiddenLayers(weights[i], out)
    end
    

    # TODO: fully connected
    sentences = Any[]
    for i in 1:size(out, 1)
        a = out[i,1,:]
        fully = a*w[5] .+ w[6]
        push!(sentences, fully)
    end

    return sentences
end

# Loss function (softmax)
# Should be adaptive softmax according to the paper
function loss(weights, input, ygold, numLayers)
    total = 0
    ypred = predict(weights, input, numLayers)
    for i in 1:size(ygold, 1)
        ynorm = logp(ypred[i], 1)
        ygold_word = ygold[i]
        total = total + sum(ygold .* ynorm) # check minus sign before sum()
    end

    y = total/size(ygold, 1)
    return y
end

lossgradient = grad(loss)

# Generates Ygold matrix for one sentence.
# The size for each YGold is vocabSize x sentence length.
# Ygold contains one-hot vectors for each sentence.
function generateYgold(sentence, vocabSize)
    yGold = Any[]
    for i in 1:size(sentence, 1)
        oneHotVec = zeros(vocabSize, 1)
        oneHotVec[i] = 1
        push!(yGold, oneHotVec)
    end
    return yGold
end

# here x is a single batch of x_batches 
# size of x is 64
# Gradient Clipping = 0.1
function train(weights, input, y, gradient_clip, embeddingMatrix, vocabSize, numHiddenLayers)
    for i in 1:size(input, 1)
        sentenceLength = size(input, 2)
        # Generate yGold of size vocabSize x number of words in a sentence (size(x,2))
        yGold = generateYgold(y[i], vocabSize)
        
        # Generate X from IDs in sentences and embedding matrix for 0th hidden Layers
        # size = sentenceSize x embeddingMatrixSize
        sentenceEmbeddings = Any[]
        for word in 1:sentenceLength
            embed = embeddingMatrix[word]
            push!(sentenceEmbeddings, embed)
        end

        # Reshape to make X 4D
        X = reshape(sentenceEmbeddings, size(sentenceEmbeddings)..., 1)

        # Loss, Gradient Clipping...
        gloss = lossgradient(weights, X, yGold, numHiddenLayers)
        gnorm = 0

        for k= 1:size(gloss, 1)
            gnorm += sumabs2(gloss[k])
        end

        gnorm = sqrt(gnorm)
        
        if gnorm > gclip
            for k = 1:size(weights,1)
                gloss[k] = (gloss[k] * gradient_clip)/gnorm
            end
        end

        params = map(x->Knet.Momentum(), weights)
        for k = 1:size(weights,1)
            update!(weights[k], gloss[k], params[k])
        end
    end
    return weights
end

# here xtst,ytst is a single batch of x_batches 
# pass Weights from training
# function accuracy(weights, xtst, ytst, embeddingMatrix, vocabulary, numLayers)
#     ncorrect = 0
#     ninstances = 0
#     for i in size(ytst, 1)
#         sentenceLength = size(input, 2)
#         yGold = generateYgold(ytst[i], size(vocabulary))
#         sentenceEmbeddings = Any[]
#         for word in 1:sentenceLength
#             embed = embeddingMatrix[word]
#             push!(sentenceEmbeddings, embed)
#         end
#         X = reshape(sentenceEmbeddings, size(sentenceEmbeddings)..., 1)
#         ypred = predict(weights, input, numLayers)
#         for j in 1:size(ytst,2)
#             ncorrect += sum(y .* (y_pred .== maximum(y_pred,1)))
#         end
#         ninstance += size(ytst,2)
#     end

#     return ncorrect
# end

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

# returns the size of a dictionary
function dictSize(dict)
    size = 0
    for i in keys(dict)
        size += 1
    end
    return size
end

main()