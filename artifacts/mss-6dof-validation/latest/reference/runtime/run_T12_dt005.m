addpath('/private/tmp/mss-6dof-reference/CRAFT/USV/models');
addpath('/private/tmp/mss-6dof-reference/GNC');
addpath('/private/tmp/mss-6dof-reference/HYDRO');
addpath('/private/tmp/mss-6dof-reference/LIBRARY/kinematics');
addpath('/private/tmp/mss-6dof-reference/LIBRARY/modeling');
addpath('/private/tmp/mss-6dof-reference/LIBRARY/numericalMethods');
addpath('/Users/aadikenchammanaold/Desktop/LEADCAT/Website/AAMAS27-Sim/artifacts/mss-6dof-validation/latest/reference/runtime');
data=dlmread('/Users/aadikenchammanaold/Desktop/LEADCAT/Website/AAMAS27-Sim/artifacts/mss-6dof-validation/latest/reference/runtime/sensitivity_T12/input.csv',',',1,0); nrows=rows(data); x=[0;0;0;0;0;0;0;0;0;0;0;0]; n=zeros(2,1);
out=zeros(nrows,21); dt=0.0050000000000000001;
for i=1:nrows
  row=data(i,:); out(i,:)=[row(1),x',n',row(2:7)];
  if i<nrows
    f=@(xx) otter(xx,n,0,zeros(3,1),0,0);
    k1=f(x); k2=f(x+dt*k1/2); k3=f(x+dt*k2/2); k4=f(x+dt*k3);
    x=x+dt*(k1+2*k2+2*k3+k4)/6;
    if 1
      nc=row(8:9)'; n=n+dt/0.1*(nc-n);
      [~,~,~,~,nmin,nmax]=otter(); n=min(max(n,nmin),nmax);
    end
  end
end
dlmwrite('/Users/aadikenchammanaold/Desktop/LEADCAT/Website/AAMAS27-Sim/artifacts/mss-6dof-validation/latest/reference/runtime/sensitivity_T12/mss_raw.csv',out,'delimiter',',','precision','%.17g');
