%---------------------------------------------------------
%
%   Program: add2path
%
%   Purpose: recursively add directory paths to my matlab path
%
%   Programmer: Rod Pickens
%
%   Date: Nov 18, 2013
%
%---------------------------------------------------------

function add2path(path2search)

  if nargin == 0
      path2search = '.';
      clc;
      fprintf(1,'\n\n\tBuilding directory tree from current location %s.\n',pwd);
  end
  
   % add hardcoded paths outside directory in path2search
  addpath('C:\als\dvemp\04_Software\code\tools\Matlab');
  addpath('C:\als\dvemp\04_Software\code\tools\Matlab\CoordLib');
  addpath('C:\als\dvemp\04_Software\code\tools\Matlab\CoordLib\bin');

  add2DaPathRecursive(path2search);

  addpath('C:\als\dvemp\04_Software\code\tools\Matlab');
  addpath('C:\als\dvemp\04_Software\code\tools\Matlab\CoordLib');
  addpath('C:\als\dvemp\04_Software\code\tools\Matlab\CoordLib\bin');
end
function add2DaPathRecursive(path2search)

    dirFlag=7;
    
    list=dir(path2search);
    for index=3:size(list)
       fprintf(1,'%s --> %s\n',pwd,list(index).name); 

       if list(index).isdir
          if strfind(sprintf('%s',list(index).name),'data')
          else
              currDir = pwd;
              path2add=[currDir filesep list(index).name];
              fprintf(1,'%s\n',path2add); 

              if ~strcmp(path,'.') && exist(path2add,'dir')==dirFlag
                   cd(path2add);
                   addpath(path2add);
                   add2DaPathRecursive(path2add);
                   cd(currDir);
              end
          end
       end
    end

return
end

% function add2path(path2search)
%     dirFlag=7;
%     
%     list=dir(path2search);
%     for index=3:size(list)
%        %fprintf(1,'%s --> %s\n',pwd,list(index).name); 
%        if list(index).isdir
%           currDir = pwd;
%           path2add=[currDir filesep list(index).name];
%           %fprintf(1,'%s\n',path2add); 
%                    
%           if ~strcmp(path,'.') && exist(path2add,'dir')==dirFlag
%                cd(path2add);
%                add2path(path2add);
%                add2path(path2add);
%                cd(currDir);
%           end
%        end
% 
%     end
% 
% end  
        
        