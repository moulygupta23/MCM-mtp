function [y,r,elem] = ReadSparse(file)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
fid=fopen(file);
str= fgetl(fid);
y=[];
r=[];
field1 = 'value';
field2 = 'ind';
value = [];
s = struct(field1,value,field2,value);
elem=0;
while ischar(str)
    tempx=[];
    tempind=[];
    rexpression = '[\s\:]';
    splitStr = regexp(str,rexpression,'split');
    n=length(splitStr);
    y=[y;str2num(splitStr{1})];
    if  strcmp(splitStr{n},'')==0
        flag=0;
        for i=2:n
            if flag==0
                ind=str2num(splitStr{i});
                tempind=[tempind,ind];
                flag=1;
            else
                tempx=[tempx,str2num(splitStr{i})];
                flag=0;
            end
        end
        %ind
        %elem
        elem=max(elem,ind);
        if isempty(elem)
            input('working?');
        end
    end
    s.value=tempx;
    s.ind=tempind;
    %x=[x;tempx];
    %ind=[ind;tempind];
    r=[r;s];
    str = fgetl(fid);
end
elem
fclose(fid);
disp('reading complete');
input('press enter');
end

