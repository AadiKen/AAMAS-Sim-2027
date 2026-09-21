addpath('/private/tmp/mss-6dof-reference/LIBRARY/modeling'); addpath('/private/tmp/mss-6dof-reference/HYDRO');
d=dlmread('/Users/aadikenchammanaold/Desktop/LEADCAT/Website/AAMAS27-Sim/artifacts/mss-6dof-validation/latest/reference/runtime/component_samples.csv',','); out=zeros(rows(d),12); G=[0 0 0 0 0 0;0 0 0 0 0 0;0 0 7541.4375 0 1508.2875000000001 0;0 0 0 1077.3296069033102 0 0;0 0 1508.2875000000001 0 2853.4339024390238 0;0 0 0 0 0 0];
for i=1:rows(d)
 eta=d(i,1:6)'; nu=d(i,7:12)';
 out(i,:)=[(-G*eta)',crossFlowDrag(2.0,0.25,0.13414634146341461,nu)'];
end
dlmwrite('/Users/aadikenchammanaold/Desktop/LEADCAT/Website/AAMAS27-Sim/artifacts/mss-6dof-validation/latest/reference/runtime/component_mss.csv',out,'delimiter',',','precision','%.17g');
