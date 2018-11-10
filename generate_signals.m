%% Script pour générer la banque des signaux
clc; clear all
n=1024;
data=zeros(n,5);
data(:,1)=loadPcwConst('rect',n);
data(:,2)=loadPcwConst('equidistant',n);
data(:,3)=loadPcwConst('sampleDec',n);
data(:,4)=loadPcwConst('sample1',n);
data(:,5)=loadPcwConst('sample2',n);
for i =1:5
    subplot(5,1,i)
    plot(data(:,i))
end

csvwrite('signalBank1024.csv',data)