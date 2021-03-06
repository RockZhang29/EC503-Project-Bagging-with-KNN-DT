%TREEC Build a decision tree classifier
% 
%   W = TREEC(A,CRIT,PRUNE,T)
% 
% Computation of a decision tree classifier out of a dataset A using 
% a binary splitting criterion CRIT:
%   INFCRIT  -  information gain
%   MAXCRIT  -  purity (default)
%   FISHCRIT -  Fisher criterion
% 
% Pruning is defined by prune:
%   PRUNE = -1 pessimistic pruning as defined by Quinlan. 
%   PRUNE = -2 testset pruning using the dataset T, or, if not
%              supplied, an artificially generated testset of 5 x size of
%              the training set based on parzen density estimates.
%              see PARZENML and GENDATP.
%   PRUNE = 0 no pruning (default).
%   PRUNE > 0 early pruning, e.g. prune = 3
%   PRUNE = 10 causes heavy pruning.

function w = treec(a,crit,prune,t)

	prtrace(mfilename);

	% When no input data is given, an empty tree is defined:
	if nargin == 0 | isempty(a)
		if nargin <2, 
			w = mapping('treec');
		elseif nargin < 3, w = mapping('treec',{crit});
		elseif nargin < 4, w = mapping('treec',{crit,prune});
		else, w = mapping('treec',{crit,prune,t});
		end
		w = setname(w,'DecTree');
		return
	end
	
	if nargin < 3, prune = []; end
	if nargin < 2, crit = []; end
	parmin_max = [1,3;-1,10];
	optcrit = inf;
	if isnan(crit) & isnan(prune)        % optimize criterion and pruning, grid search
		global REGOPT_OPTCRIT REGOPT_PARS
		for n = 1:3
			defs = {n,0};
			v = regoptc(a,mfilename,{crit,prune},defs,[2],parmin_max,testc([],'soft'),[0,0]);
			if REGOPT_OPTCRIT < optcrit
				w = v; optcrit = REGOPT_OPTCRIT; regoptpars = REGOPT_PARS;
			end
		end
		REGOPT_PARS = regoptpars;
	elseif isnan(crit)                    % optimize criterion 
		defs = {1,0};
		w = regoptc(a,mfilename,{crit,prune},defs,[1],parmin_max,testc([],'soft'),[0,0]);
	elseif isnan(prune)                    % optimize pruning
		defs = {1,0};
		w = regoptc(a,mfilename,{crit,prune},defs,[2],parmin_max,testc([],'soft'),[0,0]);
		
	else %  training for given parameters
	
		islabtype(a,'crisp');
		isvaldfile(a,1,2); % at least 1 object per class, 2 classes
		%a = testdatasize(a);
		a = dataset(a);

		% First get some useful parameters:
		[m,k,c] = getsize(a);
		nlab = getnlab(a);

		% Define the splitting criterion:
		if nargin == 1 | isempty(crit), crit = 2; end
		if ~isstr(crit)
			if crit == 0 | crit == 1, crit = 'infcrit'; 
			elseif crit == 2, crit = 'maxcrit';
			elseif crit == 3, crit = 'fishcrit';
			else, error('Unknown criterion value');
			end
		end

		% Now the training can really start:
		if (nargin == 1) | (nargin == 2)
			tree = maketree(+a,nlab,c,crit);
		elseif nargin > 2
			% We have to apply a pruning strategy:
			if prune == -1, prune = 'prunep'; end
			if prune == -2, prune = 'prunet'; end
			% The strategy can be prunep/prunet:
			if isstr(prune)
				tree = maketree(+a,nlab,c,crit);
				if prune == 'prunep'
					tree = prunep(tree,a,nlab);
				elseif prune == 'prunet'
					if nargin < 4
						t = gendatp(a,5*sum(nlab==1));
					end
					tree = prunet(tree,t);
				else
					error('unknown pruning option defined');
				end
			else
				% otherwise the tree is just cut after level 'prune'
				tree = maketree(+a,nlab,c,crit,prune);
			end
		else
			error('Wrong number of parameters')
		end

		% Store the results:
		w = mapping('tree_map','trained',{tree,1},getlablist(a),k,c);
		w = setname(w,'DecTree');
		w = setcost(w,a);
		
	end
	return
end

function tree = maketree(a,nlab,c,crit,stop) 
	prtrace(mfilename);
	[m,k] = size(a); 
	if nargin < 5, stop = 0; end;
	if nargin < 4, crit = []; end;
	if isempty(crit), crit = 'infcrit'; end;

	% Construct the tree:

	% When all objects have the same label, create an end-node:
	if all([nlab == nlab(1)]) 
		% Avoid giving 0-1 probabilities, but 'regularize' them a bit using
		% a 'uniform' Bayesian prior:
		p = ones(1,c)/(m+c); p(nlab(1)) = (m+1)/(m+c);
		tree = [nlab(1),0,0,0,p];
	else
		% now the tree is recursively constructed further:
		[f,j,t] = feval(crit,+a,nlab); % use desired split criterion
		if isempty(t)
			crt = 0;
		else
			crt = infstop(+a,nlab,j,t);    % use desired early stopping criterion
		end
		p = sum(expandd(nlab),1);
		if length(p) < c, p = [p,zeros(1,c-length(p))]; end
		% When the stop criterion is not reached yet, we recursively split
		% further:
		if crt > stop
			% Make the left branch:
			J = find(a(:,j) <= t);
			tl = maketree(+a(J,:),nlab(J),c,crit,stop);
			% Make the right branch:
			K = find(a(:,j) > t);
			tr = maketree(+a(K,:),nlab(K),c,crit,stop);
			% Fix the node labelings before the branches can be 'glued'
			% together to a big tree:
			[t1,t2] = size(tl);
			tl = tl + [zeros(t1,2) tl(:,[3 4])>0 zeros(t1,c)];
			[t3,t4] = size(tr);
			tr = tr + (t1+1)*[zeros(t3,2) tr(:,[3 4])>0 zeros(t3,c)];
			% Make the complete tree: the split-node and the branches:
			tree= [[j,t,2,t1+2,(p+1)/(m+c)]; tl; tr]; 
		else
			% We reached the stop criterion, so make an end-node:
			[mt,cmax] = max(p);
			tree = [cmax,0,0,0,(p+1)/(m+c)];
		end
	end
	return
end

function [f,j,t] = maxcrit(a,nlab)
	prtrace(mfilename);
	[m,k] = size(a);
	c = max(nlab);
	T = zeros(2*c,k); R = zeros(2*c,k);
	for j = 1:c
		L = (nlab == j);
		if sum(L) == 0
			T([2*j-1:2*j],:) = zeros(2,k);
			R([2*j-1:2*j],:) = zeros(2,k);
		else
			T(2*j-1,:) = min(a(L,:),[],1);
			R(2*j-1,:) = sum(a < ones(m,1)*T(2*j-1,:),1);
			T(2*j,:) = max(a(L,:),[],1);
			R(2*j,:) = sum(a > ones(m,1)*T(2*j,:),1);
		end
	end
	% From R the purity index for all features is computed:
	G = R .* (m-R);
	% and the best feature is found:
	[gmax,tmax] = max(G,[],1);
	[f,j] = max(gmax);
	Tmax = tmax(j);
	if Tmax ~= 2*floor(Tmax/2)
		t = (T(Tmax,j) + max(a(find(a(:,j) < T(Tmax,j)),j)))/2;
	else
		t = (T(Tmax,j) + min(a(find(a(:,j) > T(Tmax,j)),j)))/2;
	end
	if isempty(t)
		[f,j,t] = infcrit(a,nlab);
		prwarning(3,'Maxcrit not feasible for decision tree, infcrit is used')
	end
	return
end
    
function [g,j,t] = infcrit(a,nlab)
	prtrace(mfilename);
	[m,k] = size(a);
	c = max(nlab);
	mininfo = ones(k,2);
	% determine feature domains of interest
	[sn,ln] = min(a,[],1); 
	[sx,lx] = max(a,[],1);
	JN = (nlab(:,ones(1,k)) == ones(m,1)*nlab(ln)') * realmax;
	JX = -(nlab(:,ones(1,k)) == ones(m,1)*nlab(lx)') * realmax;
	S = sort([sn; min(a+JN,[],1); max(a+JX,[],1); sx]);
	% S(2,:) to S(3,:) are interesting feature domains
	P = sort(a);
	Q = (P >= ones(m,1)*S(2,:)) & (P <= ones(m,1)*S(3,:));
	% these are the feature values in those domains
	for f=1:k,		% repeat for all features
		af = a(:,f);
		JQ = find(Q(:,f));
		SET = P(JQ,f)';
		if JQ(1) ~= 1
			SET = [P(JQ(1)-1,f), SET];
		end
		n = length(JQ);
		if JQ(n) ~= m
			SET = [SET, P(JQ(n)+1,f)];
		end
		n = length(SET) -1;
		T = (SET(1:n) + SET(2:n+1))/2; % all possible thresholds
		L = zeros(c,n); R = L;     % left and right node object counts per class
		for j = 1:c
			J = find(nlab==j); mj = length(J);
			if mj == 0
				L(j,:) = realmin*ones(1,n); R(j,:) = L(j,:);
			else
				L(j,:) = sum(repmat(af(J),1,n) <= repmat(T,mj,1), 1) + realmin;
				R(j,:) = sum(repmat(af(J),1,n) > repmat(T,mj,1), 1) + realmin;
			end
		end
		infomeas =  - (sum(L .* log10(L./(ones(c,1)*sum(L, 1))), 1) ...
			       + sum(R .* log10(R./(ones(c,1)*sum(R, 1))), 1)) ...
		    ./ (log10(2)*(sum(L, 1)+sum(R, 1))); % criterion value for all thresholds
		[mininfo(f,1),j] = min(infomeas);     % finds the best
		mininfo(f,2) = T(j);     % and its threshold
	end   
	g = 1-mininfo(:,1)';
	[finfo,j] = min(mininfo(:,1));		% best over all features
	t = mininfo(j,2);			% and its threshold
	return

end

function [f,j,t] = fishcrit(a,nlab)
	prtrace(mfilename);
	[m,k] = size(a);
	c = max(nlab);
	if c > 2
		error('Not more than 2 classes allowed for Fisher Criterion')
	end
	% Get the mean and variances of both the classes:
	J1 = find(nlab==1);
	J2 = find(nlab==2);
	u = (mean(a(J1,:),1) - mean(a(J2,:),1)).^2;
	s = std(a(J1,:),0,1).^2 + std(a(J2,:),0,1).^2 + realmin;
	% The Fisher ratio becomes:
	f = u ./ s;
	% Find then the best feature:
	[ff,j] = max(f);
	% Given the feature, compute the threshold:
	m1 = mean(a(J1,j),1);
	m2 = mean(a(J2,j),1);
	w1 = m1 - m2; w2 = (m1*m1-m2*m2)/2;
	if abs(w1) < eps % the means are equal, so the Fisher
			 % criterion (should) become 0. Let us set the thresold
			 % halfway the domain
			 t = (max(a(J1,j),[],1) + min(a(J2,j),[],1)) / 2;
	else
		t = w2/w1;
	end
	return
end

function crt = infstop(a,nlab,j,t)
	prtrace(mfilename);
	[m,k] = size(a);
	c = max(nlab);
	aj = a(:,j);
	ELAB = expandd(nlab); 
	L = sum(ELAB(aj <= t,:),1) + 0.001;
	R = sum(ELAB(aj > t,:),1) + 0.001;
	LL = (L+R) * sum(L) / m;
	RR = (L+R) * sum(R) / m;
	crt = sum(((L-LL).^2)./LL + ((R-RR).^2)./RR);
	return
end

function tree = prunep(tree,a,nlab,num)
	prtrace(mfilename);
	if nargin < 4, num = 1; end;
	[N,k] = size(a);
	c = size(tree,2)-4;
	if tree(num,3) == 0, return, end;
	w = mapping('treec','trained',{tree,num},[1:c]',k,c);
	ttt=tree_map(dataset(a,nlab),w);
	J = testc(ttt)*N;
	EA = J + nleaves(tree,num)./2;   % expected number of errors in tree
	P = sum(expandd(nlab,c),1);     % distribution of classes
					%disp([length(P) c])
					[pm,cm] = max(P);     % most frequent class
					E = N - pm;     % errors if substituted by leave
					SD = sqrt((EA * (N - EA))/N);
					if (E + 0.5) < (EA + SD)	     % clean tree while removing nodes
						[mt,kt] = size(tree);
						nodes = zeros(mt,1); nodes(num) = 1; n = 0;
						while sum(nodes) > n;	     % find all nodes to be removed
							n = sum(nodes);
							J = find(tree(:,3)>0 & nodes==1);
							nodes(tree(J,3)) = ones(length(J),1); 
							nodes(tree(J,4)) = ones(length(J),1); 
						end
						tree(num,:) = [cm 0 0 0 P/N];
						nodes(num) = 0; nc = cumsum(nodes);
						J = find(tree(:,3)>0);% update internal references
						tree(J,[3 4]) = tree(J,[3 4]) - reshape(nc(tree(J,[3 4])),length(J),2);
						tree = tree(~nodes,:);% remove obsolete nodes
					else 
						K1 = find(a(:,tree(num,1)) <= tree(num,2));
						K2 = find(a(:,tree(num,1)) >  tree(num,2));

						tree = prunep(tree,a(K1,:),nlab(K1),tree(num,3));
						tree = prunep(tree,a(K2,:),nlab(K2),tree(num,4));
					end
					return
end

function tree = prunet(tree,a)
	prtrace(mfilename);
	[m,k] = size(a);
	[n,s] = size(tree);
	c = s-4;
	erre = zeros(1,n);
	deln = zeros(1,n);
	w = mapping('treec','trained',{tree,1},[1:c]',k,c);
	[f,lab,nn] = tree_map(a,w);  % bug, this works only if a is dataset, labels ???
	[fmax,cmax] = max(tree(:,[5:4+c]),[],2);
	nngood = nn([1:n]'+(cmax-1)*n);
	errn = sum(nn,2) - nngood;% errors in each node
	sd = 1;
	while sd > 0
		erre = zeros(n,1);
		deln = zeros(1,n);
		endn = find(tree(:,3) == 0)';	% endnodes
		pendl = max(tree(:,3*ones(1,length(endn)))' == endn(ones(n,1),:)');
		pendr = max(tree(:,4*ones(1,length(endn)))' == endn(ones(n,1),:)');
		pend = find(pendl & pendr);		% parents of two endnodes
		erre(pend) = errn(tree(pend,3)) + errn(tree(pend,4));
		deln = pend(find(erre(pend) >= errn(pend))); % nodes to be leaved
		sd = length(deln);
		if sd > 0
			tree(tree(deln,3),:) = -1*ones(sd,s);
			tree(tree(deln,4),:) = -1*ones(sd,s);
			tree(deln,[1,2,3,4]) = [cmax(deln),zeros(sd,3)];
		end
	end
	return
end

function number = nleaves(tree,num)
	prtrace(mfilename);
	if nargin < 2, num = 1; end
	if tree(num,3) == 0
		number = 1 ;
	else
		number = nleaves(tree,tree(num,3)) + nleaves(tree,tree(num,4));
	end
	return
end
