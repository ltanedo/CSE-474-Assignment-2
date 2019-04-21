#testing
    #print "i am here"
    #print ("n_input: ", n_input)
    #print ("n_hidden: ", n_hidden)
    # print ("n_class: ", n_class)
    # print ("W1: ", W1)
    #print ("train_data: ", train_data)
    #print ("train_label: ", train_label)
    
    # assune bias at end: 
    # add column of 1s to bias
    train_data = np.column_stack((train_data, np.ones((train_data.shape[0]))))
    train_label = np.column_stack((train_label, np.ones((train_label.shape[0]))))
    
    print(training_data)
    #print(training_data)

    #global vars 
    output_loss = np.zeros((n_class,n_class))
    hidden_loss = np.zeros((len(train_data),1))

    print("shape of input data: ",train_data.shape)
    print("shape of weight 1: ",W1.shape)

    print("shape of output data: ",train_label.shape)
    print("shape of weight 2: ",W2.shape)

    hidden_loss = sigmoid(np.matmul(W1,np.transpose(train_data)))
    print("shape of hiddenVal: ",output_loss.shape)

    output_loss = sigmoid(np.matmul(np.transpose(W2),train_label))
    print("shape of hiddenVal: ",hidden_loss.shape)



    ########################## feed foward propogation ################################

    #iteration for each train label x of i 
    # for iteration in range(n_class):

    #     for node in range(n_hidden):

    #         # hidden variables
    #         input_weights = W1[node]                # 1 x 6
    #         input_values = train_data[iteration]    # 1 x 5 <- no bias
            
    #         # hidden node losses
    #         # loss = WtX
    #         hidden_loss[iteration][node] = sigmoid(np.dot(input_weights, input_values))
                    
    #     print (node," hidden node")
    #     #print (input_values)
    #     print (hidden_loss)

    #     for node in range(n_class):
            
    #         # output variables
    #         output_weights = W2[node]
    #         output_values = hidden_loss[iteration]                                                      # possible mistake

    #         # bias correction on output_values
    #         output_values = np.append(output_values, [1], axis = 0)

    #         # output node losses
    #         # loss = WtX
    #         output_loss[iteration][node] = sigmoid(np.dot(output_weights, output_values))
            
    #         #print (node," output node")
    #         #print (output_weights)
    #         #print (output_values)
    #         #print(output_loss[iteration][node])

    # # train_label -> oneHot
    y = convertOneHot(train_label,n_class)

    # # check labels 
    # print("\n output layer")
    # print (output_loss)
    # print('\n', "train_label")
    # print(train_label)



    # ################################### JW Error ###################################



    # ones array
    # print("outputloss shape"+ output_loss.shape)
    neg_ones = np

    #Rewrite:
    third = np.log(1 - output_loss)
    second = 1 - y
    first = np.matmul(y, np.transpose(np.log(output_loss)))
    last = first + np.matmul(second,np.transpose(third))
    last = np.sum(last) / len(last) * -1
    # last = -1 * np.mean(last)

    #end Rewrite

    # ln(1 - o.il)
    # third = 1 - output_loss
    # third = np.log(third)

    # # 1 - y.il
    # second = 1 - y

    # #y.il ln o.il
    # print(y.shape)
    # print(output_loss.shape)
    # first = np.matmul(y, np.transpose(np.log(output_loss)))
    # temp = np.matmul(second, np.transpose(third))

    # print('output_loss' + str(temp.shape))

    # # final
    # obj_val = (first + temp) 
    print(last)
    # obj_val = -np.mean(obj_val)
    print("obj_val")
    print(last)
    print("\n")
    
    obj_val = np.sum(obj_val)




    ########################### BACK propogation ############################


    #You need to get two matrices, grad_W1 and grad_W2 which will have the same exact shape 
    #s the W1 and W2 matrices. Each entry of these matrices will be the partial derivative of 
    #the loss function with respect to the weight entry in the corresponding weight matrix.

    # # get W2
    # hidden_loss = np.delete(hidden_loss, np.s_[-1:], axis=1)
    # W2 = np.matmul(hidden_loss,(output_loss - y))
    # print(W2)

    # # get W1
    # print(np.shape(train_data))
    # W1 = np.matmul((1 - hidden_loss) * hidden_loss * ((output_loss - y) * W2), train_data)
    # #W1 = (1 - hidden_loss) * hidden_loss *

    # print(W1)