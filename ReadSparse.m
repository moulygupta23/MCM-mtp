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
    flag=0;
    for i=2:n
        if flag==0
            tempind=[tempind,str2num(splitStr{i})];
            flag=1;
        else
            tempx=[tempx,str2num(splitStr{i})];
            flag=0;
        end
    end
    elem=max(elem,length(tempx));
    s.value=tempx;
    s.ind=tempind;
    %x=[x;tempx];
    %ind=[ind;tempind];
    r=[r;s];
    str = fgetl(fid);
end
fclose(fid);

end

