function xyz = readVerts(filename)
xyz_name = strrep(filename,'.dat','.xyz');
%{
if( exist(xyz_name,'file') == 2 )
    try
        xyz = load(xyz_name);
        return
    catch
    end
end
%}
fid = fopen(filename);
i = 0;
while( fgetl(fid) > -1 )
    i = i + 1;
end
nl = i - 1;
xyz = zeros(nl*8,4);
frewind(fid);
i = 0;
k = 0;
while( i < nl )
    i = i + 1;
    line = fgetl(fid);
    line = strrep(line,'(','');
    dat = split(line,' ');
    trp = split(dat{2},')');
    for j = 1:(length(trp)-1)
        k = k + 1;
        xyz(k,2:4) = str2num(trp{j});
        xyz(k,1) = i;
    end
end
fclose(fid);
save(xyz_name,'xyz','-ascii');
end