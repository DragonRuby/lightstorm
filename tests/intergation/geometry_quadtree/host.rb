Hash.class_eval {
  [:x, :y, :h, :w].each do |s|
    define_method s do self[s]; end
  end
}

objects = []
50.times do |i|
  r = {
    x: rand(1280),
    y: rand(720),
    w: 16,
    h: 16,
  }
  objects << r
end

puts GTK::Geometry::quadtree_bounding_box(objects)
