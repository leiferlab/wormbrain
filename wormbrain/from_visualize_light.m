sz_stupid = size(neurons.neurons);
sz = sz_stupid(2);

stringa = "[";
for i=1:sz
    if i==1
        stringa = append(stringa,jsonencode(neurons.neurons(1,i)));
    else
        stringa = append(stringa,",",jsonencode(neurons.neurons(1,i)));
    end
end

stringa = append(stringa,"]");

