function ret = merit_funm_nonscalar_w_nonlin(ffun, gfun, rho)
	ret = ffun + rho*norm(gfun,1);
end