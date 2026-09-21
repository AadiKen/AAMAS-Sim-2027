addpath('/private/tmp/mss-6dof-reference/CRAFT/USV/models');
addpath('/private/tmp/mss-6dof-reference/GNC');
addpath('/private/tmp/mss-6dof-reference/HYDRO');
addpath('/private/tmp/mss-6dof-reference/LIBRARY/kinematics');
addpath('/private/tmp/mss-6dof-reference/LIBRARY/modeling');
addpath('/private/tmp/mss-6dof-reference/LIBRARY/numericalMethods');
addpath('/Users/aadikenchammanaold/Desktop/LEADCAT/Website/AAMAS27-Sim/artifacts/mss-6dof-validation/latest/reference/runtime');
data=dlmread('/Users/aadikenchammanaold/Desktop/LEADCAT/Website/AAMAS27-Sim/artifacts/mss-6dof-validation/latest/cases/T04_roll/input.csv',',',1,0); nrows=rows(data); x=[0;0;0;0;0;0;0;0;0;0;0;0]; n=zeros(2,1);
out=zeros(nrows,21); dt=0.01;
for i=1:nrows
  row=data(i,:); out(i,:)=[row(1),x',n',row(2:7)];
  if i<nrows
    f=@(xx) otter_tau(xx,zeros(2,1),0,zeros(3,1),0,0,row(2:7)');
    k1=f(x); k2=f(x+dt*k1/2); k3=f(x+dt*k2/2); k4=f(x+dt*k3);
    x=x+dt*(k1+2*k2+2*k3+k4)/6;
    if 0
      nc=row(8:9)'; n=n+dt/0.1*(nc-n);
      [~,~,~,~,nmin,nmax]=otter(); n=min(max(n,nmin),nmax);
    end
  end
end
dlmwrite('/Users/aadikenchammanaold/Desktop/LEADCAT/Website/AAMAS27-Sim/artifacts/mss-6dof-validation/latest/cases/T04_roll/mss_raw.csv',out,'delimiter',',','precision','%.17g');
