import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# INPUTS
input_neurons = 784
hidden_neurons = 30
output_neurons = 10
epochs = 4
mini_batch_size = 10
learning_rate = 3.0

# NETWORK
n = network.Network([input_neurons, hidden_neurons, output_neurons])

# EXECUTION
def samples_for_class(_c):
    samples = []
    for sample in test_data:
        if sample[1] == _c:
            samples.append(sample)
    return samples

avg_for_class = {}
avg = 0.0
for _c in range(0, 10):
    print("Class: {0}".format(_c))

    avg_for_class[_c] = 0.0
    samples = samples_for_class(_c)
    res = n._SGD(training_data, epochs, mini_batch_size, learning_rate, test_data=samples)
    for k in res.keys():
        avg_for_class[_c] += float(res[k][0]) / float(res[k][1])
        print "Epoch {0}: {1} / {2}".format(k, res[k][0], res[k][1])
    avg_for_class[_c] /= float(len(res))
    avg += avg_for_class[_c]
    print("Average for {0}: {1}".format(_c, avg_for_class[_c]))

avg /= 10
print("TOTAL AVERAGE: {0}".format(avg))