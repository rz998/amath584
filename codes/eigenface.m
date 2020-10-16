close all;
% Cropped Images
Z = 'CroppedYale';
Y = dir(fullfile(Z,'*'));
N = setdiff({Y([Y.isdir]).name},{'.','..'});
D=[];
for i = 1:numel(N)
    W = dir(fullfile(Z,N{i},'*'));
    X = {W(~[W.isdir]).name};
    for j = 1:numel(X)
        F = fullfile(Z,N{i},X{j});
        E=imresize(double(imread(F)),[192 168]);
        G=reshape(E,[32256,1]);
        D=[D G];
    end
end

trainD=D(:,1:2304);
[U,S,V]=svd(trainD,'econ');


figure(1)
subplot(2,3,6);semilogy(diag(S),'*')
ylim([1e+0 1e+5])
yticks([1e+0 1e+3 1e+5])
xlabel('$k$','interpreter','latex')
ylabel('Singular values $\sigma_k$','interpreter','latex')
set(gca,'Fontsize',[12])
for r=1:5 % Testing for different r values
    subplot(2,3,r);facel=reshape(U(:,r),[192 168]);imagesc(facel),colormap gray;
    set(gca,'XTick',[], 'YTick', [])
end
saveas(gcf,'eigenface.png')

figure(2)
X=D(:,2305); % First face of the 37th person
subplot(2,4,1);imagesc(reshape(X,[192 168])),colormap gray
title('Test Face','interpreter','latex')
set(gca,'XTick',[], 'YTick', [])
set(gca,'Fontsize',[12])
i=1;
for r=[25 50 100 200 400 800 1600] % Testing for different r values
    Xrec = U(:,1:r)*(U(:,1:r)'*X);
    i=i+1;
    subplot(2,4,i);imagesc(reshape(Xrec,[192 168])),colormap gray
    title(['$r=~$',num2str(r)],'interpreter','latex')
    set(gca,'XTick',[], 'YTick', [])
    set(gca,'Fontsize',[12])
end
saveas(gcf,'testface1.png')

figure(3)
X=D(:,2369); % First face of the 38th person
subplot(2,4,1);imagesc(reshape(X,[192 168])),colormap gray
title('Test Face','interpreter','latex')
set(gca,'XTick',[], 'YTick', [])
set(gca,'Fontsize',[12])
i=1;
for r=[25 50 100 200 400 800 1600] 
    Xrec = U(:,1:r)*(U(:,1:r)'*X);
    i=i+1;
    subplot(2,4,i);imagesc(reshape(Xrec,[192 168])),colormap gray
    title(['$r=~$',num2str(r)],'interpreter','latex')
    set(gca,'XTick',[], 'YTick', [])
    set(gca,'Fontsize',[12])
end
saveas(gcf,'testface2.png')


% Uncropped Images
Z = 'yalefaces';
W = dir(fullfile(Z,'*'));
X = {W(~[W.isdir]).name};
D1=[];
for i = 1:numel(X)
    F = fullfile(Z,X{i});
    E=imresize(double(imread(F)),[192 168]);
    G=reshape(E,[32256,1]);
    D1=[D1 G];
end

trainD1=D1(:,1:143);
[U1,S1,V1]=svd(trainD1,'econ');

figure(4)
X=D1(:,144); 
subplot(2,4,1);imagesc(reshape(X,[192 168])),colormap gray
title('Test Face','interpreter','latex')
set(gca,'XTick',[], 'YTick', [])
set(gca,'Fontsize',[12])
i=1;
for r=[20 40 60 80 100 120 140] 
    Xrec = U1(:,1:r)*(U1(:,1:r)'*X);
    i=i+1;
    subplot(2,4,i);imagesc(reshape(Xrec,[192 168])),colormap gray
    title(['$r=~$',num2str(r)],'interpreter','latex')
    set(gca,'XTick',[], 'YTick', [])
    set(gca,'Fontsize',[12])
end
saveas(gcf,'testface3.png')


figure(5)
X=D1(:,155); 
subplot(2,4,1);imagesc(reshape(X,[192 168])),colormap gray
title('Test Face','interpreter','latex')
set(gca,'XTick',[], 'YTick', [])
set(gca,'Fontsize',[12])
i=1;
for r=[20 40 60 80 100 120 140] 
    Xrec = U1(:,1:r)*(U1(:,1:r)'*X);
    i=i+1;
    subplot(2,4,i);imagesc(reshape(Xrec,[192 168])),colormap gray
    title(['$r=~$',num2str(r)],'interpreter','latex')
    set(gca,'XTick',[], 'YTick', [])
    set(gca,'Fontsize',[12])
end
saveas(gcf,'testface4.png')

figure(6)
semilogy(diag(S(1:143,1:143)),'*')
ylim([1e+2 1e+6])
yticks([1e+2 1e+4 1e+6])
hold on
semilogy(diag(S1),'o')
xlabel('$k$','interpreter','latex')
ylabel('Singular values $\sigma_k$','interpreter','latex')
legend('Cropped Images','Uncropped Images','interpreter','latex')
set(gca,'TickLabelInterpreter', 'tex');
set(gca,'Fontsize',[12])
saveas(gcf,'decay.png')



