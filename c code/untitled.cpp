static void solve_c_svc(
	const svm_problem *prob, const svm_parameter* param,
	double *alpha, Solver::SolutionInfo* si, double Cp, double Cn)
{
	int l = prob->l;
	double *minus_ones = new double[l];
	schar *y = new schar[l];

	int i;

	for(i=0;i<l;i++)
	{
		alpha[i] = 0;
		minus_ones[i] = -1;
		if(prob->y[i] > 0) y[i] = +1; else y[i] = -1;
	}

	Solver s;
	s.Solve(l, SVC_Q(*prob,*param,y) , minus_ones, y,
		alpha, Cp, Cn, param->eps, si, param->shrinking);
	{
		//Solve(int l, const QMatrix& Q, const double *p_, const schar *y_,
		   // double *alpha_, double Cp, double Cn, double eps,
		   // SolutionInfo* si, int shrinking)
		QMatrix& Q=SVC_Q(*prob,*param,y);
		{ in SVC_Q
			clone(y,y_,prob.l);
			cache = new Cache(prob.l,(long int)(param.cache_size*(1<<20)));
			QD = new double[prob.l];
			for(int i=0;i<prob.l;i++)
				QD[i] = (this->*kernel_function)(i,i);
		}
		this->l = l;
		this->Q = &Q;
		QD=Q.get_QD();
		clone(p, p_,l);
		clone(y, y_,l);
		clone(alpha,alpha_,l);
		this->Cp = Cp;
		this->Cn = Cn;
		this->eps = eps;
		unshrink = false;

		// initialize alpha_status
		{
			alpha_status = new char[l];
			for(int i=0;i<l;i++){
				if(alpha[i] >= get_C(i))
					alpha_status[i] = UPPER_BOUND;
				else if(alpha[i] <= 0)
					alpha_status[i] = LOWER_BOUND;
				else alpha_status[i] = FREE;
			}
		}

		// initialize active set (for shrinking)
		{
			active_set = new int[l];
			for(int i=0;i<l;i++)
				active_set[i] = i;
			active_size = l;
		}

		// initialize gradient
		{
			G = new double[l];
			G_bar = new double[l];
			int i;
			for(i=0;i<l;i++)
			{
				G[i] = p[i];
				G_bar[i] = 0;
			}
			for(i=0;i<l;i++)
				if(!is_lower_bound(i))
				{
					const Qfloat *Q_i = Q.get_Q(i,l);
					{in get_Q
						Qfloat *data;
						int start, j;
						if((start = cache->get_data(i,&data,len)) < len)
						{
							for(j=start;j<len;j++)
								data[j] = (Qfloat)(y[i]*y[j]*(this->*kernel_function)(i,j));
						}
						return data;
					}
					double alpha_i = alpha[i];
					int j;
					for(j=0;j<l;j++)
						G[j] += alpha_i*Q_i[j];
					if(is_upper_bound(i))
						for(j=0;j<l;j++)
							G_bar[j] += get_C(i) * Q_i[j];
				}
		}

		// optimization step

		int iter = 0;
		int max_iter = max(10000000, l>INT_MAX/100 ? INT_MAX : 100*l);
		int counter = min(l,1000)+1;
		
		while(iter < max_iter)
		{
			// show progress and do shrinking

			if(--counter == 0)
			{
				counter = min(l,1000);
				if(shrinking) do_shrinking();
				{in do_shrinking
					int i;
					double Gmax1 = -INF;		// max { -y_i * grad(f)_i | i in I_up(\alpha) }
					double Gmax2 = -INF;		// max { y_i * grad(f)_i | i in I_low(\alpha) }

					// find maximal violating pair first
					for(i=0;i<active_size;i++)
					{
						if(y[i]==+1)	
						{
							if(!is_upper_bound(i))	
							{
								if(-G[i] >= Gmax1)
									Gmax1 = -G[i];
							}
							if(!is_lower_bound(i))	
							{
								if(G[i] >= Gmax2)
									Gmax2 = G[i];
							}
						}
						else	
						{
							if(!is_upper_bound(i))	
							{
								if(-G[i] >= Gmax2)
									Gmax2 = -G[i];
							}
							if(!is_lower_bound(i))	
							{
								if(G[i] >= Gmax1)
									Gmax1 = G[i];
							}
						}
					}

					if(unshrink == false && Gmax1 + Gmax2 <= eps*10) 
					{
						unshrink = true;
						reconstruct_gradient();
						{in reconstruct_gradient
							// reconstruct inactive elements of G from G_bar and free variables

							if(active_size == l) return;

							int i,j;
							int nr_free = 0;

							for(j=active_size;j<l;j++)
								G[j] = G_bar[j] + p[j];

							for(j=0;j<active_size;j++)
								if(is_free(j))
									nr_free++;

							if(2*nr_free < active_size)
								info("\nWARNING: using -h 0 may be faster\n");

							if (nr_free*l > 2*active_size*(l-active_size))
							{
								for(i=active_size;i<l;i++)
								{
									const Qfloat *Q_i = Q->get_Q(i,active_size);
									for(j=0;j<active_size;j++)
										if(is_free(j))
											G[i] += alpha[j] * Q_i[j];
								}
							}
							else
							{
								for(i=0;i<active_size;i++)
									if(is_free(i))
									{
										const Qfloat *Q_i = Q->get_Q(i,l);
										double alpha_i = alpha[i];
										for(j=active_size;j<l;j++)
											G[j] += alpha_i * Q_i[j];
									}
							}
						}
						active_size = l;
						info("*");
					}

					for(i=0;i<active_size;i++)
						if (be_shrunk(i, Gmax1, Gmax2))
							{
								if(is_upper_bound(i))
								{
									if(y[i]==+1)
										return(-G[i] > Gmax1);
									else
										return(-G[i] > Gmax2);
								}
								else if(is_lower_bound(i))
								{
									if(y[i]==+1)
										return(G[i] > Gmax2);
									else	
										return(G[i] > Gmax1);
								}
								else
									return(false);								
							}
						{
							active_size--;
							while (active_size > i)
							{
								if (!be_shrunk(active_size, Gmax1, Gmax2))
									{
										if(is_upper_bound(i))
										{
											if(y[i]==+1)
												return(-G[i] > Gmax1);
											else
												return(-G[i] > Gmax2);
										}
										else if(is_lower_bound(i))
										{
											if(y[i]==+1)
												return(G[i] > Gmax2);
											else	
												return(G[i] > Gmax1);
										}
										else
											return(false);										
									}
								{
									swap_index(i,active_size);
									break;
								}
								active_size--;
							}
						}
					}
				info(".");
			}

			int i,j;
			if(select_working_set(i,j)!=0)
				{in select_working_set
					{
						// return i,j such that
						// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
						// j: minimizes the decrease of obj value
						//    (if quadratic coefficeint <= 0, replace it with tau)
						//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)
						
						double Gmax = -INF;
						double Gmax2 = -INF;
						int Gmax_idx = -1;
						int Gmin_idx = -1;
						double obj_diff_min = INF;

						for(int t=0;t<active_size;t++)
							if(y[t]==+1)	
							{
								if(!is_upper_bound(t))
									if(-G[t] >= Gmax)
									{
										Gmax = -G[t];
										Gmax_idx = t;
									}
							}
							else
							{
								if(!is_lower_bound(t))
									if(G[t] >= Gmax)
									{
										Gmax = G[t];
										Gmax_idx = t;
									}
							}

						int i = Gmax_idx;
						const Qfloat *Q_i = NULL;
						if(i != -1) // NULL Q_i not accessed: Gmax=-INF if i=-1
							Q_i = Q->get_Q(i,active_size);

						for(int j=0;j<active_size;j++)
						{
							if(y[j]==+1)
							{
								if (!is_lower_bound(j))
								{
									double grad_diff=Gmax+G[j];
									if (G[j] >= Gmax2)
										Gmax2 = G[j];
									if (grad_diff > 0)
									{
										double obj_diff;
										double quad_coef = QD[i]+QD[j]-2.0*y[i]*Q_i[j];
										if (quad_coef > 0)
											obj_diff = -(grad_diff*grad_diff)/quad_coef;
										else
											obj_diff = -(grad_diff*grad_diff)/TAU;

										if (obj_diff <= obj_diff_min)
										{
											Gmin_idx=j;
											obj_diff_min = obj_diff;
										}
									}
								}
							}
							else
							{
								if (!is_upper_bound(j))
								{
									double grad_diff= Gmax-G[j];
									if (-G[j] >= Gmax2)
										Gmax2 = -G[j];
									if (grad_diff > 0)
									{
										double obj_diff;
										double quad_coef = QD[i]+QD[j]+2.0*y[i]*Q_i[j];
										if (quad_coef > 0)
											obj_diff = -(grad_diff*grad_diff)/quad_coef;
										else
											obj_diff = -(grad_diff*grad_diff)/TAU;

										if (obj_diff <= obj_diff_min)
										{
											Gmin_idx=j;
											obj_diff_min = obj_diff;
										}
									}
								}
							}
						}

						if(Gmax+Gmax2 < eps)
							return 1;

						out_i = Gmax_idx;
						out_j = Gmin_idx;
						return 0;
					}	
				}
			{
				// reconstruct the whole gradient
				reconstruct_gradient();
				{in reconstruct_gradient
					// reconstruct inactive elements of G from G_bar and free variables

					if(active_size == l) return;

					int i,j;
					int nr_free = 0;

					for(j=active_size;j<l;j++)
						G[j] = G_bar[j] + p[j];

					for(j=0;j<active_size;j++)
						if(is_free(j))
							nr_free++;

					if(2*nr_free < active_size)
						info("\nWARNING: using -h 0 may be faster\n");

					if (nr_free*l > 2*active_size*(l-active_size))
					{
						for(i=active_size;i<l;i++)
						{
							const Qfloat *Q_i = Q->get_Q(i,active_size);
							for(j=0;j<active_size;j++)
								if(is_free(j))
									G[i] += alpha[j] * Q_i[j];
						}
					}
					else
					{
						for(i=0;i<active_size;i++)
							if(is_free(i))
							{
								const Qfloat *Q_i = Q->get_Q(i,l);
								double alpha_i = alpha[i];
								for(j=active_size;j<l;j++)
									G[j] += alpha_i * Q_i[j];
							}
				}
					
				// reset active set size and check
				active_size = l;
				info("*");
				if(select_working_set(i,j)!=0)
					break;
				else
					counter = 1;	// do shrinking next iteration
			}
			
			++iter;

			// update alpha[i] and alpha[j], handle bounds carefully
			
			const Qfloat *Q_i = Q.get_Q(i,active_size);//order m
			const Qfloat *Q_j = Q.get_Q(j,active_size);

			double C_i = get_C(i);
			double C_j = get_C(j);

			double old_alpha_i = alpha[i];
			double old_alpha_j = alpha[j];

			if(y[i]!=y[j])
			{
				double quad_coef = QD[i]+QD[j]+2*Q_i[j];
				if (quad_coef <= 0)
					quad_coef = TAU;
				double delta = (-G[i]-G[j])/quad_coef;
				double diff = alpha[i] - alpha[j];
				alpha[i] += delta;
				alpha[j] += delta;
				
				if(diff > 0)
				{
					if(alpha[j] < 0)
					{
						alpha[j] = 0;
						alpha[i] = diff;
					}
				}
				else
				{
					if(alpha[i] < 0)
					{
						alpha[i] = 0;
						alpha[j] = -diff;
					}
				}
				if(diff > C_i - C_j)
				{
					if(alpha[i] > C_i)
					{
						alpha[i] = C_i;
						alpha[j] = C_i - diff;
					}
				}
				else
				{
					if(alpha[j] > C_j)
					{
						alpha[j] = C_j;
						alpha[i] = C_j + diff;
					}
				}
			}
			else
			{
				double quad_coef = QD[i]+QD[j]-2*Q_i[j];
				if (quad_coef <= 0)
					quad_coef = TAU;
				double delta = (G[i]-G[j])/quad_coef;
				double sum = alpha[i] + alpha[j];
				alpha[i] -= delta;
				alpha[j] += delta;

				if(sum > C_i)
				{
					if(alpha[i] > C_i)
					{
						alpha[i] = C_i;
						alpha[j] = sum - C_i;
					}
				}
				else
				{
					if(alpha[j] < 0)
					{
						alpha[j] = 0;
						alpha[i] = sum;
					}
				}
				if(sum > C_j)
				{
					if(alpha[j] > C_j)
					{
						alpha[j] = C_j;
						alpha[i] = sum - C_j;
					}
				}
				else
				{
					if(alpha[i] < 0)
					{
						alpha[i] = 0;
						alpha[j] = sum;
					}
				}
			}

			// update G

			double delta_alpha_i = alpha[i] - old_alpha_i;
			double delta_alpha_j = alpha[j] - old_alpha_j;
			
			for(int k=0;k<active_size;k++)
			{
				G[k] += Q_i[k]*delta_alpha_i + Q_j[k]*delta_alpha_j;
			}

			// update alpha_status and G_bar

			{
				bool ui = is_upper_bound(i);
				bool uj = is_upper_bound(j);
				update_alpha_status(i);
				{in update_alpha_status
					if(alpha[i] >= get_C(i))
						alpha_status[i] = UPPER_BOUND;
					else if(alpha[i] <= 0)
						alpha_status[i] = LOWER_BOUND;
					else alpha_status[i] = FREE;
				}
				update_alpha_status(j);
				int k;
				if(ui != is_upper_bound(i))
				{
					Q_i = Q.get_Q(i,l);
					if(ui)
						for(k=0;k<l;k++)
							G_bar[k] -= C_i * Q_i[k];
					else
						for(k=0;k<l;k++)
							G_bar[k] += C_i * Q_i[k];
				}

				if(uj != is_upper_bound(j))
				{
					Q_j = Q.get_Q(j,l);
					if(uj)
						for(k=0;k<l;k++)
							G_bar[k] -= C_j * Q_j[k];
					else
						for(k=0;k<l;k++)
							G_bar[k] += C_j * Q_j[k];
				}
			}
		}

		if(iter >= max_iter)
		{
			if(active_size < l)
			{
				// reconstruct the whole gradient to calculate objective value
				reconstruct_gradient();
				{in reconstruct_gradient
					// reconstruct inactive elements of G from G_bar and free variables

					if(active_size == l) return;

					int i,j;
					int nr_free = 0;

					for(j=active_size;j<l;j++)
						G[j] = G_bar[j] + p[j];

					for(j=0;j<active_size;j++)
						if(is_free(j))
							nr_free++;

					if(2*nr_free < active_size)
						info("\nWARNING: using -h 0 may be faster\n");

					if (nr_free*l > 2*active_size*(l-active_size))
					{
						for(i=active_size;i<l;i++)
						{
							const Qfloat *Q_i = Q->get_Q(i,active_size);
							for(j=0;j<active_size;j++)
								if(is_free(j))
									G[i] += alpha[j] * Q_i[j];
						}
					}
					else
					{
						for(i=0;i<active_size;i++)
							if(is_free(i))
							{
								const Qfloat *Q_i = Q->get_Q(i,l);
								double alpha_i = alpha[i];
								for(j=active_size;j<l;j++)
									G[j] += alpha_i * Q_i[j];
							}
				}
				active_size = l;
				info("*");
			}
			fprintf(stderr,"\nWARNING: reaching max number of iterations\n");
		}

		// calculate rho

		si->rho = calculate_rho();

		// calculate objective value
		{
			double v = 0;
			int i;
			for(i=0;i<l;i++)
				v += alpha[i] * (G[i] + p[i]);

			si->obj = v/2;
		}

		// put back the solution
		{
			for(int i=0;i<l;i++)
				alpha_[active_set[i]] = alpha[i];
		}

		// juggle everything back
		/*{
			for(int i=0;i<l;i++)
				while(active_set[i] != i)
					swap_index(i,active_set[i]);
					// or Q.swap_index(i,active_set[i]);
		}*/

		si->upper_bound_p = Cp;
		si->upper_bound_n = Cn;

		info("\noptimization finished, #iter = %d\n",iter);

		delete[] p;
		delete[] y;
		delete[] alpha;
		delete[] alpha_status;
		delete[] active_set;
		delete[] G;
		delete[] G_bar;
	}

	double sum_alpha=0;
	for(i=0;i<l;i++)
		sum_alpha += alpha[i];

	if (Cp==Cn)
		info("nu = %f\n", sum_alpha/(Cp*prob->l));

	for(i=0;i<l;i++)
		alpha[i] *= y[i];

	delete[] minus_ones;
	delete[] y;
}