close all; clear all; clc;
import casadi.*

% --- ENVIRONMENT SETUP ---
env_size = [50, 50];
num_obstacles = 20;
min_spacing = 0.8;
min_clearance = 2.0;

% --- SEED SETUP ---
E0_center = [25; 25];
E0_axes = [0.75, 0.375];   % Half of 1.5 x 0.75
E0_rot = pi/6;
R = [cos(E0_rot), -sin(E0_rot); sin(E0_rot), cos(E0_rot)];
E0.L = R * diag(E0_axes);
E0.d = E0_center;

% --- CONVEX POLYTOPE SEED AS RECTANGLE (1.5 x 0.75) ---
w = 1.5; h = 0.75;
rect_local = 0.5 * [ -w -h;
                     w -h;
                     w  h;
                    -w  h ];
Q = (R * rect_local')' + E0_center';
Q_hull = convhull(Q(:,1), Q(:,2));

% --- ELLIPSE POINTS FOR COLLISION CHECK ---
theta_e = linspace(0, 2*pi, 100);
ellipse_pts = (E0.L * [cos(theta_e); sin(theta_e)]) + E0.d;

% --- GENERATE OBSTACLES INSIDE 6x6 SEED BOUNDING SQUARE ---
bounding_square = [E0_center(1)-3, E0_center(2)-3;
                   E0_center(1)+3, E0_center(2)+3];

convex_obstacles = generate_nonoverlapping_obstacles(env_size, num_obstacles, ...
    min_spacing, Q, ellipse_pts, min_clearance, bounding_square);

% --- PLOTTING ---
figure; hold on; axis equal; grid on;
xlim([0, env_size(1)]);
ylim([0, env_size(2)]);
xlabel('X (m)'); ylabel('Y (m)');
title('FIRI Seed Initialization with Bounding Box and Rectangle Polytope');

% Plot bounding box for reference
rectangle('Position', [bounding_square(1,1), bounding_square(1,2), 6, 6], ...
    'EdgeColor', [0.2 0.2 1], 'LineStyle', '--', 'LineWidth', 1.5);

% Plot obstacles
for i = 1:length(convex_obstacles)
    obs = convex_obstacles{i};
    if size(unique(obs, 'rows'), 1) >= 3
        idx = convhull(obs(:,1), obs(:,2));
        fill(obs(idx,1), obs(idx,2), [0.5 0.5 0.5], ...
            'FaceAlpha', 0.5, 'EdgeColor', 'k', 'HandleVisibility', 'off');
    end
end

% Plot seed polytope Q
fill(Q(Q_hull,1), Q(Q_hull,2), 'w', 'EdgeColor', 'g', ...
     'FaceColor', 'none', 'LineWidth', 2, 'DisplayName', '$\mathcal{Q}$');
plot(Q(:,1), Q(:,2), 'go', 'MarkerSize', 6, ...
     'MarkerFaceColor', 'g', 'HandleVisibility', 'off');

% Plot ellipsoid ε₀
draw_ellipse(E0.L, E0.d, '$\varepsilon_0$', 'r');

% Legend
legend('Interpreter', 'latex', 'Location', 'northeast');
hold off;

% --- FUNCTION DEFINITIONS ---
function convex_obs = generate_nonoverlapping_obstacles(env_size, num_obs, min_spacing, Q, ellipse_pts, min_dist, box)
    convex_obs = cell(num_obs,1);
    placed_centers = zeros(num_obs,2);
    Q_hull = convhull(Q(:,1), Q(:,2));
    Q_hull_pts = Q(Q_hull, :);

    for i = 1:num_obs
        num_pts = randi([3,8]);
        is_valid = false;
        attempts = 0;

        while ~is_valid && attempts < 100
            center = [rand*(box(2,1)-box(1,1)) + box(1,1), ...
                      rand*(box(2,2)-box(1,2)) + box(1,2)];
            attempts = attempts + 1;
            angles = linspace(0, 2*pi, num_pts+1)' + rand*pi/4;
            angles(end) = [];
            radii = rand(num_pts,1)*2 + 1;
            x_obs = center(1) + radii .* cos(angles);
            y_obs = center(2) + radii .* sin(angles);
            x_obs = min(max(x_obs,0), env_size(1));
            y_obs = min(max(y_obs,0), env_size(2));
            obs = [x_obs, y_obs];

            if (i == 1 || all(vecnorm(placed_centers(1:i-1,:) - center,2,2) > min_spacing)) && ...
                is_polygon_clear(obs, Q_hull_pts, ellipse_pts, min_dist)
                is_valid = true;
                convex_obs{i} = obs;
                placed_centers(i,:) = center;
            end
        end
    end
end

function is_far = is_polygon_clear(obs, Q_hull_pts, ellipse_pts, min_dist)
    dists_to_Q = pdist2(obs, Q_hull_pts);
    dists_to_E = pdist2(obs, ellipse_pts');
    is_far = all(min(dists_to_Q,[],2) > min_dist) && all(min(dists_to_E,[],2) > min_dist);
end

function h = draw_ellipse(L_draw, d_draw, dispName, color)
    theta = linspace(0, 2*pi, 100);
    circle = [cos(theta); sin(theta)];
    ellipse_pts = L_draw * circle + d_draw;
    h = plot(ellipse_pts(1,:), ellipse_pts(2,:), '-', ...
             'Color', color, 'LineWidth', 2, 'DisplayName', dispName);
end
