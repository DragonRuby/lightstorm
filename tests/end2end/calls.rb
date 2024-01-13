def add x, y
  x + y
end

module M
  def f
    42
  end
end

puts add(2, M.f)
