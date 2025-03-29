function FIRI_main
    % FIRI_main.m - Robust SOCP implementation for FIRI algorithm
    close all; clear; clc;
    import casadi.*
    
    %% Environment parameters
    env_size = [50, 50];          % 50m x 50m environment
    num_obstacles = 40;           % Number of convex obstacles
    min_spacing = 0.1;            % Minimum spacing between obstacles
    ellipsoid_spacing = 0.5;      % Spacing from ellipsoid seed
    
    %% Seed ellipsoid parameters
    AE = 1;                     % No rotation
    DE = diag([2, 1]);          % Semi-axis lengths
    bE = [15; 25];              % Center coordinates
    
    % Generate seed ellipsoid and convex hull
    Q = generate_ellipsoid_seed(AE, DE, bE);
    P_seed = convhull(Q(:,1), Q(:,2));
    
    %% Generate obstacles
    convex_obstacles = generate_nonoverlapping_obstacles(env_size, num_obstacles,...
        min_spacing, Q, ellipsoid_spacing);
    
    %% Initialize plot
    figure; hold on; grid on; axis equal;
    xlim([0, env_size(1)]); ylim([0, env_size(2)]);
    xlabel('X (m)'); ylabel('Y (m)');
    title('FIRI 2D Algorithm with Robust SOCP Implementation');
    
    % Plot obstacles
    for i = 1:numel(convex_obstacles)
        obs = convex_obstacles{i};
        if size(unique(obs, 'rows'), 1) >= 3
            k = convhull(obs(:,1), obs(:,2));
            fill(obs(k,1), obs(k,2), 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'k');
        end
    end
    
    % Plot seed elements
    plot(Q(P_seed,1), Q(P_seed,2), 'b-', 'LineWidth', 2);
    plot(Q(:,1), Q(:,2), 'go', 'MarkerSize', 6, 'MarkerFaceColor', 'g');
    
    %% Run FIRI algorithm with improved parameters
    seed_point = bE;
    [Pk, E_final, iter] = FIRI(seed_point, convex_obstacles, eye(2)*0.1, seed_point, 1e-4, 15);
    disp(['FIRI converged in ' num2str(iter) ' iterations.']);
    
    %% Plot results
    if ~isempty(Pk)
        vertices = hrep2vrep(Pk.A', Pk.b);
        if ~isempty(vertices)
            k = convhull(vertices(:,1), vertices(:,2));
            plot(vertices(k,1), vertices(k,2), 'm-', 'LineWidth', 2);
        else
            warning('Final polytope vertices could not be computed.');
        end
    end
    draw_ellipse(E_final.L, E_final.d);
    hold off;

    %% Nested Functions %%
    function [Pk, E_final, iter] = FIRI(seed_point, obstacles, L0, d0, tol, max_iters)
        E.L = L0;  E.d = d0;
        iter = 0;
        vol_prev = det(E.L);
        success_flag = true;
        
        while iter < max_iters && success_flag
            iter = iter + 1;
            
            % RsI Module
            [A, b] = RsI(E, obstacles, seed_point);
            Pk = struct('A', A, 'b', b);
            
            % Seed containment check
            if ~isempty(A) && any(A' * seed_point >= b - 1e-6)
                disp('Seed containment violated!');
                success_flag = false;
                break;
            end
            
            % MVIE Module with robust SOCP
            [L_new, d_new, success] = MVIE_SOCP(Pk, E.L, E.d);
            
            if ~success
                warning(['MVIE optimization failed at iteration ' num2str(iter)]);
                success_flag = false;
                break;
            end
            
            E_new.L = L_new;  E_new.d = d_new;
            vol_new = det(L_new);
            
            % Convergence check
            if abs(vol_new - vol_prev)/vol_prev < tol
                E_final = E_new;
                break;
            end
            E = E_new;
            vol_prev = vol_new;
        end
        E_final = E;
    end

    function [A, b] = RsI(E, obstacles, seed_point)
        A_list = [];
        b_list = [];
        for i = 1:length(obstacles)
            obs = obstacles{i};
            if size(unique(obs, 'rows'), 1) < 3, continue; end
            
            x_star = ClosestPointOnPolygon(seed_point, obs);
            [a, bi] = TangentPlane(seed_point, x_star);
            
            A_list = [A_list, a];
            b_list = [b_list; bi];
        end
        A = A_list;
        b = b_list;
    end

    function [L_new, d_new, success] = MVIE_SOCP(P, L_init, d_init)
        import casadi.*
        n = 2;
        A = P.A;
        b_vec = P.b;
        num_con = length(b_vec);
        
        opti = casadi.Opti();
        
        % Decision variables - Cholesky factor (lower triangular)
        L = opti.variable(n,n);
        d = opti.variable(n,1);
        
        % Objective: Maximize logdet(L) = 2*sum(log(diag(L)))
        obj = -2*sum(log(diag(L)));  % Minimize negative logdet
        
        % Regularization term to improve numerical stability
        reg_weight = 1e-4;
        reg_term = reg_weight*(sumsqr(L) + sumsqr(d));
        
        opti.minimize(obj + reg_term);
        
        % L must be lower triangular with positive diagonal
        opti.subject_to(L(1,2) == 0);
        opti.subject_to(L(1,1) >= 1e-4);
        opti.subject_to(L(2,2) >= 1e-4);
        
        % Obstacle avoidance constraints
        for i = 1:num_con
            ai = A(:,i);
            bi = b_vec(i);
            opti.subject_to(norm(L*ai) <= bi - ai'*d - 1e-6);  % Small buffer
        end
        
        % Initialization with previous solution
        opti.set_initial(L, L_init);
        opti.set_initial(d, d_init);
        
        % Solver options
        opts = struct('print_time', 0, 'ipopt', struct('print_level', 0));
        opti.solver('ipopt', opts);
        
        try
            sol = opti.solve();
            L_new = sol.value(L);
            d_new = sol.value(d);
            success = true;
        catch e
            warning(['MVIE_SOCP failed: ' e.message]);
            L_new = L_init;
            d_new = d_init;
            success = false;
        end
    end

    %% Helper Functions %%
    function [a, bi] = TangentPlane(seed, x_star)
        a = (x_star - seed)/norm(x_star - seed);  % Normalized normal vector
        epsilon = 1e-3;  % Increased safety margin
        bi = a'*x_star + epsilon;
    end

    function x_star = ClosestPointOnPolygon(seed, poly)
        num_pts = size(poly,1);
        min_dist = inf;
        x_star = poly(1,:)';
        for j = 1:num_pts
            p1 = poly(j,:)';
            p2 = poly(mod(j, num_pts)+1,:)';
            v = p2 - p1;
            if norm(v) < eps, continue; end
            t = max(0, min(1, ((seed - p1)' * v) / (v' * v)));
            proj = p1 + t*v;
            d = norm(seed - proj);
            if d < min_dist
                min_dist = d;
                x_star = proj;
            end
        end
    end

    function vertices = hrep2vrep(A, b)
        tol = 1e-6;
        m = size(A,1);
        pts = [];
        for i = 1:m-1
            for j = i+1:m
                M = [A(i,:); A(j,:)];
                if rank(M) < 2, continue; end
                x_int = M \ [b(i); b(j)];
                if all(A * x_int <= b + tol)
                    pts = [pts; x_int'];
                end
            end
        end
        if ~isempty(pts)
            pts = unique(pts, 'rows');
            center = mean(pts,1);
            angles = atan2(pts(:,2)-center(2), pts(:,1)-center(1));
            [~, order] = sort(angles);
            vertices = pts(order,:);
        else
            vertices = [];
        end
    end

    function draw_ellipse(L, d)
        theta = linspace(0, 2*pi, 100);
        circle = [cos(theta); sin(theta)];
        ellipse = L * circle + d;
        plot(ellipse(1,:), ellipse(2,:), 'g-', 'LineWidth', 2);
    end

    function Q = generate_ellipsoid_seed(AE, DE, bE)
        theta = linspace(0, 2*pi, 100)';
        x = [cos(theta), sin(theta)]';
        Q = (AE * DE * x)';
        Q = Q + bE';
    end

    function convex_obstacles = generate_nonoverlapping_obstacles(env_size, num_obs, min_spacing, Q, ellipsoid_spacing)
        convex_obstacles = cell(num_obs, 1);
        placed_centers = zeros(num_obs, 2);
        for i = 1:num_obs
            num_pts = randi([3, 8]);
            is_valid = false;
            while ~is_valid
                center = [rand * env_size(1), rand * env_size(2)];
                if (i == 1 || all(vecnorm(placed_centers(1:i-1,:) - center, 2, 2) > min_spacing)) && ...
                   all(vecnorm(Q - center, 2, 2) > ellipsoid_spacing)
                    is_valid = true;
                    placed_centers(i, :) = center;
                    angles = linspace(0, 2*pi, num_pts+1)' + rand*pi/4;
                    angles(end) = [];
                    radii = rand(num_pts, 1) * 2 + 1;
                    x_obs = center(1) + radii .* cos(angles);
                    y_obs = center(2) + radii .* sin(angles);
                    x_obs = min(max(x_obs, 0), env_size(1));
                    y_obs = min(max(y_obs, 0), env_size(2));
                    convex_obstacles{i} = [x_obs, y_obs];
                end
            end
        end
    end
end
