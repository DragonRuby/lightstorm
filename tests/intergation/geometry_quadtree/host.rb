Hash.class_eval {
  [:x, :y, :h, :w, :bounding_box, :rects, :top_left, :top_right, :bottom_left, :bottom_right].each do |s|
    define_method s do self[s]; end
    s_eql = (s.to_s + '=').to_sym
    define_method s_eql do |x| self[s] = x; end
  end
}

objects = []
5000.times do |i|
  r = {
    x: rand(1280),
    y: rand(720),
    w: 16,
    h: 16,
  }
  objects << r
end

tree = GTK::Geometry::quadtree_create(objects)
puts tree.size
