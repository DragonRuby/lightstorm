module GTK
  module Geometry
    class << self

      def inside_rect?(inner_rect, outer_rect)
        return false if (!inner_rect or !outer_rect)
        return false if (inner_rect.x < outer_rect.x)
        return false if (inner_rect.x + inner_rect.w > outer_rect.x + outer_rect.w)
        return false if (inner_rect.y < outer_rect.y)
        return false if (inner_rect.y + inner_rect.h > outer_rect.y + outer_rect.h)
        true
      end

      def quadtree_bounding_box rects
        return { x: 0, y: 0, w: 0, h: 0 } if !rects || rects.length == 0
        min_x = rects.first.x
        min_y = rects.first.y
        max_x = rects.first.x + rects.first.w
        max_y = rects.first.y + rects.first.h
        i = 0
        while i < rects.size
          r = rects[i]
          min_x = r.x if r.x < min_x
          min_y = r.y if r.y < min_y
          max_x = r.x + r.w if (r.x + r.w) > max_x
          max_y = r.y + r.w if (r.y + r.w) > max_y
          i += 1
        end

        { x: min_x, y: min_y, w: max_x - min_x, h: max_y - min_y }
      end

      def quadtree_insert_rect node, rect
        return if !inside_rect? rect, node.bounding_box

        node.top_left ||= {
          bounding_box: { x: node.bounding_box.x,
                          y: node.bounding_box.y + node.bounding_box.h / 2,
                          w: node.bounding_box.w / 2,
                          h: node.bounding_box.h / 2 },
          rects: []
        }

        node.top_right ||= {
          bounding_box: { x: node.bounding_box.x + node.bounding_box.w / 2,
                          y: node.bounding_box.y + node.bounding_box.h / 2,
                          w: node.bounding_box.w / 2,
                          h: node.bounding_box.h / 2 },
          rects: []
        }

        node.bottom_left ||= {
          bounding_box: { x: node.bounding_box.x,
                          y: node.bounding_box.y,
                          w: node.bounding_box.w / 2,
                          h: node.bounding_box.h / 2 },
          rects: []
        }

        node.bottom_right ||= {
          bounding_box: { x: node.bounding_box.x + node.bounding_box.w / 2,
                          y: node.bounding_box.y,
                          w: node.bounding_box.w / 2,
                          h: node.bounding_box.h / 2 },
          rects: []
        }

        if inside_rect? rect, node.top_left.bounding_box
          quadtree_insert_rect node.top_left, rect
        elsif inside_rect? rect, node.top_right.bounding_box
          quadtree_insert_rect node.top_right, rect
        elsif inside_rect? rect, node.bottom_left.bounding_box
          quadtree_insert_rect node.bottom_left, rect
        elsif inside_rect? rect, node.bottom_right.bounding_box
          quadtree_insert_rect node.bottom_right, rect
        else
          node.rects << rect
        end
      end

      def quadtree_create rects
        tree = {
          bounding_box: (quadtree_bounding_box rects),
          rects: []
        }

        i = 0
        while i < rects.size
          rect = rects[i]
          quadtree_insert_rect tree, rect
          i += 1
        end

        tree
      end


      #alias_method :quad_tree_bounding_box, :quadtree_bounding_box
      #alias_method :quad_tree_insert_rect,  :quadtree_insert_rect
      #alias_method :quad_tree_create,       :quadtree_create
      #alias_method :create_quad_tree,       :quadtree_create
      #alias_method :create_quadtree,        :quadtree_create
    end
  end
end
