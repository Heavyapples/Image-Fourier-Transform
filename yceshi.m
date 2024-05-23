clc;clear;close all; % 清空命令窗口、清空 Matlab 工作空间、关闭所有已打开的图形窗口
x=imread('D:\bixiu\x\小作业材料\ImageSet\fft.tif'); % 读取图像  
subplot(1,3,1); % 分成一行三列，第一列显示原始图像
imshow(x); % 显示原始图像  
title('原始图像'); % 设置标题
f = im_fft2(x); % 对图像进行傅里叶变换
fre = fftshift(abs(f/40)); % 计算频谱图像
subplot(1, 3, 2); % 分成一行三列，第二列显示傅里叶变换后的频谱
imshow(uint8(fre)) % 显示频谱图像
y = im_ifft2(f); % 对图像进行傅里叶反变换
subplot(1, 3, 3); % 分成一行三列，第三列显示傅里叶反变换后的图像
imshow(y) % 显示反变换后的图像
% x=double(x);                  %转换为double类型，方便离散傅里叶变换  
% [m,n]=size(x);                %获取行和列  
% M=ceil(log2(m));                  %大于log2(m) 的最小整数 
% N=ceil(log2(n));                  %大于log2(n) 的最小整数
% x1=zeros(2^M-m,n);
% x=cat(1,x,x1);                    %进行补零，使矩阵为2^M行，n列
% x2=zeros(2^M,2^N-n);           
% x=cat(2,x,x2);                    %进行补零，使矩阵为2^M行，2^N列,至此矩阵x为2^M行，2^N列
% 
% mp=0:2^M-1;                       %开始对行倒序,
% mp1=mp;
% mb=zeros(1,2^M);
% for t=1:M
%     mp2=floor(mp1/2);
%     mb=mb*2+(mp1-2*mp2);
%     mp1=mp2;
% end  
% x(:,:)=x(mb+1,:);
% np=0:2^N-1;                        %开始对列倒序
% np1=np;
% nb=zeros(1,2^N);
% for t=1:N
%     np2=floor(np1/2);
%     nb=nb*2+(np1-2*np2);
%     np1=np2;
% end
% x(:,:)=x(:,nb+1);                  %对矩阵倒序结束
% 
% t=0:2^(M-1)-1;                     %生成行旋转因子
% mw(t+1)=exp(-2*pi*1i*t/2^M);
% t=0:2^(N-1)-1;                     %生成列旋转因子
% nw(t+1)=exp(-2*pi*1i*t/2^N);
% 
% for L=1:M                          %对行进行fft变换
%     B=2^(L-1);
%     for J=0:B-1
%         P=J*2^(M-L);
%         for k=J+1:2^L:2^M
%             T1=mw(P+1)*(x(k+B,:));
%             x(k+B,:)=x(k,:)-T1;
%             x(k,:)=x(k,:)+T1;
%         end
%     end
%                
% end   
                                                   
% for L=1:N                          %对列进行fft变换
%     B=2^(L-1);
%     for J=0:B-1
%         P=J*2^(N-L);
%         for k=J+1:2^L:2^N 
%             T1=nw(P+1)*(x(:,k+B));
%             x(:,k+B)=x(:,k)-T1;
%             x(:,k)=x(:,k)+T1;
%         end
%     end             
%  end   

function f = im_fft2(x, mrows, ncols) 
% 对二维矩阵进行傅里叶变换
% x: 输入的二维矩阵
% mrows, ncols: 可选参数，指定变换结果的行数和列数

if ismatrix(x) % 判断输入的是否为二维矩阵
    if nargin==1 % 如果未指定行数和列数，则使用默认值
        f = fftn(x);
    else
        f = fftn(x,[mrows ncols]); % 对矩阵进行FFT变换
    end
else % 输入不是二维矩阵
    if nargin==1 % 如果未指定行数和列数，则使用默认值
        f = fft(fft(x,[],2),[],1); % 对矩阵进行傅里叶变换
    else
        f = fft(fft(x,ncols,2),mrows,1); % 对矩阵进行傅里叶变换
    end
end   
end


function x = im_ifft2(f,varargin)
% 对二维矩阵进行反傅里叶变换
% f: 输入的二维矩阵
% varargin: 可选参数，用于指定结果行数、列数和对称性

narginchk(1,4) % 检查输入参数个数是否正确

% 获取输入矩阵的行数和列数，并将输入参数的个数和对称性（默认为 'nonsymmetric'）
% 存储在变量 num_inputs 和 symmetry 中。
m_in = size(f, 1);
n_in = size(f, 2);
num_inputs = nargin;
symmetry = 'nonsymmetric';

%检查是否指定了反变换的对称性。如果输入参数个数大于 1，且最后一个输入参数是字符串或字符数组，
% 则将其作为反变换的对称性，将 num_inputs 减 1。
if num_inputs > 1 && (isstring(varargin{end}) || ischar(varargin{end}))
    symmetry = varargin{end};
    num_inputs = num_inputs - 1;
end

% 根据输入参数的个数指定反变换结果的行数和列数。如果输入参数个数为 1，则将输出矩阵的行数和列数设置为输入矩阵的行数和列数。
% 如果输入参数个数为 2，则抛出一个异常。
% 如果输入参数个数为 3，则将输出矩阵的行数和列数设置为前两个输入参数。否则，抛出另一个异常。
if num_inputs == 1
    m_out = m_in;
    n_out = n_in;
elseif num_inputs == 2
    error(message('MATLAB:ifft2:invalidSyntax')) 
elseif num_inputs == 3
    m_out = double(varargin{1});
    n_out = double(varargin{2});
else
    error(message('MATLAB:ifft2:InvalidTrailingStringArgument'));
end

% 如果输入矩阵不是单精度或双精度浮点数，则将其转换为双精度浮点数。
if ~isfloat(f)
    f = double(f);
end

% 如果输出矩阵的行数和列数不等于输入矩阵的行数和列数，
% 则使用 `zeros` 函数创建一个新的矩阵 `f2`，大小等于输出矩阵的行数和列数，
% 并将其类型设置为与输入矩阵相同的类型。
% 然后，将输入矩阵的一部分复制到 `f2` 中，并将 `f2` 赋值给 `f`。
if m_out ~= m_in || n_out ~= n_in
    out_size = size(f);
    out_size(1) = m_out;
    out_size(2) = n_out;
    f2 = zeros(out_size, class(f));
    mm = min(m_out, m_in);
    nn = min(n_out, n_in);
    f2(1:mm, 1:nn, :) = f(1:mm, 1:nn, :);
    f = f2;
end

% 如果输入矩阵是一个二维矩阵，则调用 ifftn 函数对其进行反傅里叶变换。
% 如果输入矩阵是一个多维矩阵，则先使用 ifft 函数对其进行行方向的反傅里叶变换，
% 再对结果进行列方向的反傅里叶变换。最后一个参数是反变换的对称性，
% 如果指定了对称性，则使用指定的对称性，否则默认为 'nonsymmetric'。
if ismatrix(f)
    x = ifftn(f, symmetry);
else
    x = ifft(ifft(f, [], 2), [], 1, symmetry);
end   
end