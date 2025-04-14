function FIRI_Complete
    % FIRI_Complete.m - Complete implementation with all helper functions
    close all; clear; clc;
    import casadi.*
    
    %% Environment Setup
    env_size = [50, 50];          % Environment dimensions [m]
    num_obstacles = 30;           % Number of obstacles
    min_spacing = 0.8;            % Minimum obstacle spacing
    min_clearance = 2.0;          % Clearance from seed
    bounding_box_size = 6;        % Bounding box half-size

    %% Seed Configuration
    E0_center = [25; 25];         % Seed center coordinates
    w = 1.5; h = 0.75;            % Rectangle dimensions
    E0_rot = pi/6;                % Rotation angle
    R = [cos(E0_rot), -sin(E0_rot); sin(E0_rot), cos(E0_rot)];
    
    % Generate seed rectangle
    rect_local = 0.5 * [ -w -h; w -h; w h; -w h ];
    Q = (R * rect_local')' + E0_center';
    Q_hull = convhull(Q(:,1), Q(:,2));
    P_seed = Q(Q_hull,:);
    
    % Initial ellipsoid parameters
    E0.L = R * diag([w/2, h/2]);  % Semi-axes matrix
    E0.d = E0_center;             % Center

    %% Obstacle Generation
    bounding_square = [E0_center(1)-bounding_box_size, E0_center(2)-bounding_box_size;
                       E0_center(1)+bounding_box_size, E0_center(2)+bounding_box_size];
    
    convex_obstacles = generate_obstacles(env_size, num_obstacles,...
        min_spacing, Q, E0, min_clearance, bounding_square);

    %% Main Algorithm Execution
    [Pk, E_final, iter] = FIRI(E0.d, convex_obstacles, E0.L, E0.d, 1e-4, 15);
    disp(['FIRI converged in ' num2str(iter) ' iterations.']);

    %% Visualization
    figure; hold on; grid on; axis equal;
    xlim([0, env_size(1)]); ylim([0, env_size(2)]);
    xlabel('X (m)'); ylabel('Y (m)');
    title('FIRI Algorithm: Complete Implementation');
    
    % Plot bounding box
    rectangle('Position', [bounding_square(1,1), bounding_square(1,2),...
        2*bounding_box_size, 2*bounding_box_size],...
        'EdgeColor', [0.2 0.2 1], 'LineStyle', '--', 'LineWidth', 1.5,...
        'HandleVisibility', 'off');
    
    % Plot obstacles
    for i = 1:numel(convex_obstacles)
        obs = convex_obstacles{i};
        if size(unique(obs, 'rows'), 1) >= 3
            k = convhull(obs(:,1), obs(:,2));
            fill(obs(k,1), obs(k,2), [0.5 0.5 0.5], 'FaceAlpha', 0.5,...
                'EdgeColor', 'k', 'HandleVisibility', 'off');
        end
    end
    
    % Plot seed elements
    h_seed = plot(P_seed([1:end 1],1), P_seed([1:end 1],2), 'g-', 'LineWidth', 2);
    
    % Draw initial ellipsoid (red)
    h_initial = draw_ellipse(E0.L, E0.d, 'r');
    
    % Plot final results
    h_final_poly = gobjects(1); % Initialize graphics object
    if ~isempty(Pk)
        vertices = hrep2vrep(Pk.A', Pk.b);
        if ~isempty(vertices)
            k = convhull(vertices(:,1), vertices(:,2));
            h_final_poly = plot(vertices(k,1), vertices(k,2), 'm-',...
                'LineWidth', 2, 'DisplayName', 'Final Polytope');
        end
    end
    
    % Draw final ellipsoid (blue)
    h_final_ellipse = draw_ellipse(E_final.L, E_final.d, 'b');
    
    % Create legend (handle empty graphics objects gracefully)
    valid_handles = [h_seed, h_initial, h_final_ellipse];
    if ~isempty(h_final_poly)
        valid_handles = [valid_handles, h_final_poly];
    end
    
    legend(valid_handles, {'Seed Polytope','ε₀','ε_{final}', 'Final Polytope'},...
        'Location', 'northeast');
    hold off;
end

%% Core Algorithm Functions
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
            warning('Seed containment violated!');
            success_flag = false;
            break;
        end
        
        % MVIE Module
        [L_new, d_new, success] = MVIE_SOCP(Pk, E.L, E.d);
        
        if ~success
            warning('MVIE optimization failed at iteration %d', iter);
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
    L = opti.variable(n,n);
    d = opti.variable(n,1);
    
    % Objective function with regularization
    obj = -2*sum(log(diag(L))) + 1e-4*(sumsqr(L) + sumsqr(d));
    opti.minimize(obj);
    
    % Constraints
    opti.subject_to(L(1,2) == 0);
    opti.subject_to(L(1,1) >= 1e-4);
    opti.subject_to(L(2,2) >= 1e-4);
    
    for i = 1:num_con
        ai = A(:,i);
        bi = b_vec(i);
        opti.subject_to(norm(L*ai) <= bi - ai'*d - 1e-6);
    end
    
    opti.set_initial(L, L_init);
    opti.set_initial(d, d_init);
    
    opts = struct('print_time',0, 'ipopt',struct('print_level',0));
    opti.solver('ipopt', opts);
    
    try
        sol = opti.solve();
        L_new = sol.value(L);
        d_new = sol.value(d);
        success = true;
    catch
        L_new = L_init;
        d_new = d_init;
        success = false;
    end
end

%% Helper Functions
function convex_obs = generate_obstacles(env_size, num_obs, min_spacing, Q, E0, min_dist, box)
    convex_obs = cell(num_obs,1);
    placed_centers = zeros(num_obs,2);
    Q_hull = convhull(Q(:,1), Q(:,2));
    Q_hull_pts = Q(Q_hull,:);
    
    % Generate ellipse points for collision checking
    theta_e = linspace(0, 2*pi, 100);
    ellipse_pts = (E0.L * [cos(theta_e); sin(theta_e)]) + E0.d;

    for i = 1:num_obs
        num_pts = randi([3,8]);
        is_valid = false;
        attempts = 0;
        
        while ~is_valid && attempts < 100
            center = [rand*(box(2,1)-box(1,1)) + box(1,1),...
                      rand*(box(2,2)-box(1,2)) + box(1,2)];
            attempts = attempts + 1;
            
            % Generate random convex polygon
            angles = linspace(0, 2*pi, num_pts+1)' + rand*pi/4;
            angles(end) = [];
            radii = rand(num_pts,1)*2 + 1;
            x_obs = center(1) + radii .* cos(angles);
            y_obs = center(2) + radii .* sin(angles);
            x_obs = min(max(x_obs,0), env_size(1));
            y_obs = min(max(y_obs,0), env_size(2));
            obs = [x_obs, y_obs];
            
            % Collision checks
            dist_to_Q = min(pdist2(obs, Q_hull_pts), [], 'all');
            dist_to_E = min(pdist2(obs, ellipse_pts'), [], 'all');
            existing_dists = vecnorm(placed_centers(1:i-1,:) - center, 2, 2);
            
            if (i == 1 || all(existing_dists > min_spacing)) && ...
               dist_to_Q > min_dist && dist_to_E > min_dist
                is_valid = true;
                convex_obs{i} = obs;
                placed_centers(i,:) = center;
            end
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

function [a, bi] = TangentPlane(seed, x_star)
    a = (x_star - seed)/norm(x_star - seed);
    epsilon = 1e-3;
    bi = a'*x_star + epsilon;
end

function h = draw_ellipse(L, d, color)
    theta = linspace(0, 2*pi, 100);
    circle = [cos(theta); sin(theta)];
    ellipse = L * circle + d;
    h = plot(ellipse(1,:), ellipse(2,:), '-', 'Color', color, 'LineWidth', 2);
end
