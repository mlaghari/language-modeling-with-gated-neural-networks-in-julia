using Knet,AutoGrad,ArgParse,Compat

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--datafiles"; nargs='+'; help="If provided, use first file for training, second for dev, others for test.")
        ("--generate"; arg_type=Int; default=500; help="If non-zero generate given number of characters.")
        ("--hidden";  arg_type=Int; default=256; help="Sizes of one or more LSTM layers.")
        ("--epochs"; arg_type=Int; default=3; help="Number of epochs for training.")
        ("--batchsize"; arg_type=Int; default=100; help="Number of sequences to train on in parallel.")
        ("--seqlength"; arg_type=Int; default=25; help="Number of steps to unroll the network for.")
        ("--decay"; arg_type=Float64; default=0.9; help="Learning rate decay.")
        ("--lr"; arg_type=Float64; default=1e-1; help="Initial learning rate.")
        ("--gclip"; arg_type=Float64; default=3.0; help="Value to clip the gradient norm at.")
        ("--winit"; arg_type=Float64; default=0.1; help="Initial weights set to winit*randn().")
        ("--gcheck"; arg_type=Int; default=0; help="Check N random gradients.")
        ("--seed"; arg_type=Int; default=38; help="Random number seed.")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
    end
    return parse_args(s;as_symbols = true)        
end

##########################################################
###  CHARACTER BASED LANGUAGE MODEL BY USING RNNs      ###
###  !!!!read the comments carefullly. !!!!            ###
###  In this lab you will complete the implementation  ###
###  iRNN structure. The only difference between the   ###
###  vanilla RNN and iRNN is the way of initialization ###
###  of its hidden to hidden weight matrix.(See T 2-1) ###
###  You can complete this lab by adding less than 40  ###
###  lines of julia code!                              ###
###  Important: Name the file you send as username.jl  ###                          
##########################################################

function main(args=ARGS)
    opts = parse_commandline()
    println("opts=",[(k,v) for (k,v) in opts]...)
    opts[:seed] > 0 && srand(opts[:seed])
    opts[:atype] = eval(parse(opts[:atype]))    
    # read text and report lengths
    # path = map((@compat readstring), opts[:datafiles])
    # !isempty(path) && info("Chars read: $(map((f,c)->(basename(f),length(c)),opts[:datafiles],path))")
    path = opts[:datafiles]
    # println(string("Path is : ", path[1])) # check nargs of paths in parse_commandline
    # Task-1: Create dictionary by completing createVocabulary function
    vocab = wordOccurences(path[1], 3)

    info("$(length(vocab)) unique chars.") # The output should be 65 for input.txt 
    
    #Task-2: Create weights,parameters for Adam optimizer and initial state
    # 2-1: Weights
    # model = initweights(opts[:atype], opts[:hidden], length(vocab), opts[:winit])
    # # 2-2: Optimizer parameters
    # prms  = initparams(model)
    # # 2-3: Initial State
    # state = initstate(opts[:atype],opts[:hidden],opts[:batchsize])

    # Print some chars with randomly initialized model    
    # if opts[:generate] > 0
    #     println("########## RANDOM MODEL OUTPUT ############")        
    #     gstate = initstate(opts[:atype],opts[:hidden],1)
    #     generate(model, gstate, vocab, opts[:generate])
    # end
    
    # Task-3: Create batches :
    # We provide minibatch function for you.
    # You do not have to do anything for this step.
    # data =  map(t->minibatch(t, vocab, opts[:batchsize]), text)

    # # Print the loss of randomly initialized model.
    # losses = map(d->loss(model,copy(state),d), data) 
    # println((:epoch,0,:loss,losses...))
    
    # MAIN LOOP
    # for epoch=1:opts[:epochs]
    #     # Task-7: Implement train function to train the model. 
    #     @time train(model,prms,copy(state),data[1];slen = opts[:seqlength],lr = opts[:lr],gclip = opts[:gclip])
    #     # Calculate and print the losses after each epoch
    #     losses = map(d->loss(model,copy(state),d),data)
    #     println((:epoch,epoch,:loss,losses...))
    #     # gradcheck for testing
    #     if opts[:gcheck] > 0
    #         gradcheck(loss, model, copy(state), data[1], 1:opts[:seqlength]; gcheck=opts[:gcheck], verbose=true)
    #     end
    # end
    ####  Training finishes here ####
    # Task-8 Implement generate function to generate some text
    # Print some chars with randomly initialized model
    # if opts[:generate] > 0
    #     println("########## FINAL  MODEL OUTPUT ############")
    #     state = initstate(opts[:atype],opts[:hidden],1)
    #     generate(model, state, vocab, opts[:generate])
    # end    
end

########################## TASK-1 ##########################
# createVocabulary takes text::Array{Any,1} that contains the
# names of datafiles you provided by  --datafiles argument.
# It returns vocabulary::Dict{Char,Int}()  for given text.
# In this lab, your text array is length of 1. For example
# if you run function with julia lab5.jl --datafiles input.txt
# then text is ["input.txt"]. Note that for the sake of simplicity,
# we do *NOT* use validation or test dataset in this lab.
# You can try it by splitting  your data into 3 different set after
# the lab. 
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



########################## TASK-2.1 ##########################
# initweights creates the weights and biases for the model.
# Model is the iRNN. ( The only difference between regular RNNs
# and iRNNs is the way of initialization of its hidden to hidden
# weight matrix. In iRNN we initialize hidden to hidden weight as
# identity matrix. All the other things are exactly the same as
# regular  RNNs.The  formula for the hidden sate of RNN is:
# h_t = tanh(h_(t-1) * Whh .+ Bhh + x_t * Wxh .+ Bxh). You
# also need to create weight and bias matrices for output
# layer as you did in previous labs. Note that we do NOT use
# embedding layer here. You may get slightly different
# results than we put in below based on in which order
# you initialize the weights of model. Don't worry about it.
# function initweights(atype, hidden, vocab, winit)
#     param = Dict()
#     # your code starts here
#     # Weights matrices
#     param["Whh"] = eye(hidden)
#     param["Wxh"] = randn(vocab, hidden)*winit
#     param["Wyh"] = randn(hidden, vocab)*winit
#     param["Bhh"] = zeros(Float32, 1, hidden)
#     param["Bxh"] = zeros(Float32, 1, hidden)
#     param["Byh"] = zeros(Float32, 1, vocab)
#     # your code ends here 
#     for k in keys(param); param[k] = convert(atype, param[k]); end
#     return param
# end

########################## TASK-2.2 ##########################
# initparams creates parameters for Adam Optimizer for each
# weight/bias matrix you created in previous step
# you can check documentation for it. Use default parameters.
# http://denizyuret.github.io/Knet.jl/latest/reference.html#Knet.Adam
# function initparams(weights)
#     prms = Dict()
#     for k in keys(weights)
#         # Your code starts here
#         prms[k] = Adam()
#         # Your code ends here
#     end
#     return prms
# end

########################## TASK-2.3 ##########################
# At each time step, we take the hidden state from previous
# time step as input. To be able to do that,first we need
# to initialize hidden state. We also store updated hidden
# states in array created here.
# We initialize state as a zero matrix.
# function initstate(atype, hidden, batchsize)
#     state = Array(Any,1)
#     # your code starts here
#     state[1] = zeros(batchsize, hidden)
#     # your code ends here 
#     return map(s->convert(atype,s), state)
# end

# 
########################## TASK-3 ##########################
# We provide minibatch function for you. You do not have to
# do it for this lab. But we suggest you to understand the
# idea since you need to do it in your own project and
# future labs 
# function minibatch(chars, char_to_index, batch_size)
#     nbatch = div(length(chars), batch_size)
#     vocab_size = length(char_to_index)
#     data = [ falses(batch_size, vocab_size) for i=1:nbatch ] # using BitArrays
#     cidx = 0
#     for c in chars            # safest way to iterate over utf-8 text
#         idata = 1 + cidx % nbatch
#         row = 1 + div(cidx, nbatch)
#         row > batch_size && break
#         col = char_to_index[c]
#         data[idata][row,col] = 1
#         cidx += 1
#     end
#     return map(d->convert(KnetArray{Float32},d), data)
# end

########################## TASK-4 ##########################
# rnn is a function that takes w  created in
# initweights, s created in initstate and input
# x whose size is batchsize x vocabulary
# You need to implement the following formula:
# h_t = tanh(h_(t-1) * Whh .+ Bhh + x_t * Wxh .+ Bxh)  
# function rnn(w,s,x)
#     # Your code starts here
#     return tanh(s[1]*w["Whh"] .+ w["Bhh"] + x*w["Wxh"] .+ w["Bxh"])
#     # Your code ends here
# end

########################## TASK-5 ##########################
# Predict is a function that predicts the output given
# the input x (size of batchsize x vocabulary size)
# and weights and state. You will call the RNN function
# you implemented at previous step, and then implement
# the final output layer here. The return should be a matrix
# and its size should be batch_size x vocabulary size.
# function predict(w, s, x)
#     # Your code starts here
#     s[1] = rnn(w,s,x)
#     return s[1] * w["Wyh"] .+ w["Byh"]
#     # Your code ends here
# end

########################## TASK-6 ##########################
# Parameters of loss function:
# w: weights you created with initweights function
# s: state   you created with initstate function
# sequence: all the batches you crerated with
# minibatch function. Its size is 11153 for input.txt
# file. each element of it is a size of
# batchsize x vocabulary matrix.
# range: range is the number you specified with
# --seqlen argument. You will use it to track
# which batches you need to use. We set it inside
# the train function for you. In loss function
# you will use the batches whose indices lays on that range. 
# You will use the predict function you implement at
# previous step to get predictions and calculate the
# loss based on your predictions and gold labels. 
# function loss(w,s,sequence,range=1:length(sequence)-1)
#     total = 0.0; count = 0
#     input = sequence[first(range)]
#     atype = typeof(AutoGrad.getval(w["Whh"]))
#     for t in range
#         # your code starts here
#         ypred = predict(w,s,input)
#         ynorm = logp(ypred,2) #ypred .- log(sum(exp(ypred),2))
#         ygold = convert(atype,sequence[t+1])
#         total += sum(ygold .* ynorm)
#         count += size(ygold,1)
#         input = ygold
#         # your code ends here 
#     end
#     return -total / count
# end

# Knet magic 
# lossgradient = grad(loss)

########################## TASK-7 ##########################
# This is the main function you will use to train your model
# It takes weights (model), optimizer parameters( prms),
# hidden state (state), all the batches(sequence),
# sequence length that defines how many time step
# we enroll the RNN(slen), learning rate(lr)
# and gradient clip(gclip)
# function train(model,prms, state, sequence; slen=25, lr=1.0, gclip=0.0)
#     # Move over all data by length of slen steps. 
#     for t = 1:slen:length(sequence)-slen
#         # Take the range  and give it to loss gradient        
#         range = t:t+slen-1
#         gloss = lossgradient(model, state, sequence, range)

#         # Task-7.0: Calculate norms of gradients. You need to
#         # iterate over all gradients given by lossgradient
#         # take sum of square of them and store in gnorm variable
#         # Finally, you need to take the square root of gnorm
#         gnorm = 0
#         # norm calculation starts here
#         for k in keys(model)
#             gnorm = gnorm + sumabs2(gloss[k])
#         end
#         gnorm = sqrt(gnorm)
#         # norm calculation ends here 
        
#         # Task-7-1: Apply gradient clip if your calculated
#         # gnorm is bigger than gradient clip you provided
#         # by the --gclip argument. To apply, it the required
#         # formula is the following:
#         # gloss[k] = (gloss[k] * gclip ) / gnorm
        
#         # gradient clip starts here
#         if gnorm >gclip
#             for k in keys(model)
#                 gloss[k] = (gloss[k] * gclip ) / gnorm
#             end
#         end
#         # gradient clip ends here
    
#         # Task-7.2: Modify weights of the model according to your
#         # gradients and adam parameters. you may want to look at
#         # update! function in Knet documentation

#         #update starts here 
#         for k in keys(model)
#             update!(model[k], gloss[k], prms[k])
#         end
#         # update ends here

        
#         # The following is needed in case AutoGrad
#         # boxes state values during gradient calculation
#         # Please do not remove following 4 lines 
#         isa(state,Vector{Any}) || error("State should not be Boxed.")
#         for i = 1:length(state)
#             state[i] = AutoGrad.getval(state[i])
#         end
#     end
# end

############### TASK-8(OPTIONAL) ####################
# Generate function is a function we use to create
# some text that is similar to our training data.
# We provide sample function to you. You can predict
# the next character by using sample function once  you
# calculate the probabilities given the input.
# index to char is the same dictionary as you created
# with createdictionary function but it works in the
# reverse direction. It gives you the character given the
# index.
# function generate(param, state,vocab, nchar )
#     index_to_char = Array(Char, length(vocab))
#     for (k,v) in vocab; index_to_char[v] = k; end
#     # Here, our batch size is 1. 
#     input = oftype(state[1], zeros(1,length(vocab)))
#     index = 1
#     for t in 1:nchar
#         # get the scores from the rnn by using predict function
#         # here your x is input variable defined above.
#         # Also, make sure now all of the elements inside
#         # input vector is zero
#         # Code starts here
#         ypred = predict(param, state, input)
#         # Code ends here 

#         # Index of predicted character given the input 
#         index = sample(exp(logp(ypred)))

#         # print next character and set the corresponding
#         # element of input to 1 

#         #Code starts here

#         #Code ends here
#     end
#     println()
# end

# function sample(p)
#     p = convert(Array,p)
#     r = rand()
#     for c = 1:length(p)
#         r -= p[c]
#         r < 0 && return c
#     end
# end

main()