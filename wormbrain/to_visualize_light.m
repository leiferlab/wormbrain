sz_stupid = size(neurons.neurons);
old_n_neurons = sz_stupid(2);

n_additional_neurons = new_n_neurons - old_n_neurons;
neu = neurons.neurons;
for j=1:n_additional_neurons-1
    neu = [neurons.neurons,neurons.neurons(1)]
end
neurons.neurons = neu

for i=1:new_n_neurons
    neurons.neurons(1,i).annotation = neurons_decoded(i).annotation;
    if i>old_n_neurons
        neurons.neurons(1,i).position(1) = neurons_decoded(i).position(1);
        neurons.neurons(1,i).position(2) = neurons_decoded(i).position(2);
        neurons.neurons(1,i).position(3) = neurons_decoded(i).position(3);
        neurons.neurons(1,i).color = [0.0,0.0,0.0,0.0];
        neurons.neurons(1,i).color_readout = [0.0,0.0,0.0,0.0];
        neurons.neurons(1,i).baseline = [0.0,0.0,0.0,0.0];
        neurons.neurons(1,i).covariance = [0.0,0.0,0.0;0.0,0.0,0.0;0.0,0.0,0.0];
        neurons.neurons(1,i).truncation = 0;
        neurons.neurons(1,i).aligned_xyzRGB = [0.0,0.0,0.0,0.0,0.0,0.0];
        neurons.neurons(1,i).outlier = [];
        neurons.neurons(1,i).deterministic_id = "";
        neurons.neurons(1,i).is_annotation_on = NaN;
        neurons.neurons(1,i).annotation_confidence = -1;
    end
end
