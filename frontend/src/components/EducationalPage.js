import React from 'react';
import './EducationalPage.css';

function EducationalPage({ onBack }) {
  return (
    <div className="educational-page">
      <div className="educational-container">
        <button className="back-button" onClick={onBack}>
          ← Back to App
        </button>

        <header className="educational-header">
          <h1>The Four Color Theorem</h1>
          <p className="subtitle">Understanding the Mathematics Behind Automatic Coloring</p>
        </header>

        <section className="educational-section">
          <h2>What is the Four Color Theorem?</h2>
          <div className="content-block">
            <p>
              The <strong>Four Color Theorem</strong> is a famous mathematical theorem that states:
            </p>
            <blockquote>
              "Any planar graph can be colored with at most four colors such that no two adjacent vertices share the same color."
            </blockquote>
            <p>
              In simpler terms, this means that for any map or drawing where regions share boundaries, 
              you can always color it using just four colors without any two adjacent regions having the same color.
            </p>
          </div>
        </section>

        <section className="educational-section">
          <h2>Historical Background</h2>
          <div className="content-block">
            <ul className="timeline">
              <li>
                <strong>1852:</strong> Francis Guthrie first proposed the problem while coloring a map of England.
              </li>
              <li>
                <strong>1879:</strong> Alfred Kempe published what he thought was a proof, but it was later found to be incorrect.
              </li>
              <li>
                <strong>1890:</strong> Percy Heawood discovered the flaw in Kempe's proof and proved the Five Color Theorem.
              </li>
              <li>
                <strong>1976:</strong> Kenneth Appel and Wolfgang Haken provided the first correct proof using computer assistance, 
                checking 1,936 special cases. This was controversial at the time as it was the first major theorem proven with computer help.
              </li>
              <li>
                <strong>1996-2005:</strong> The proof was simplified and verified, with a more formal proof completed in 2005.
              </li>
            </ul>
          </div>
        </section>

        <section className="educational-section">
          <h2>How Our Application Works</h2>
          <div className="content-block">
            <p>
              Our application uses the Four Color Theorem to automatically color images. Here's how the process works:
            </p>
          </div>
        </section>

        <section className="educational-section">
          <h3>Step 1: Image Preprocessing</h3>
          <div className="content-block">
            <p>
              The input image is first preprocessed to prepare it for analysis:
            </p>
            <ul>
              <li><strong>Resizing:</strong> Large images are downscaled to improve processing speed while maintaining quality.</li>
              <li><strong>Line Art Conversion:</strong> Optional conversion to line art mode enhances edge detection and region separation.</li>
              <li><strong>Noise Reduction:</strong> Image filters help clean up artifacts and improve region detection accuracy.</li>
            </ul>
          </div>
        </section>

        <section className="educational-section">
          <h3>Step 2: Region Detection</h3>
          <div className="content-block">
            <p>
              The application identifies distinct regions in the image that need to be colored separately. 
              We use two main approaches:
            </p>
            <div className="algorithm-box">
              <h4>Traditional Method (Connected Components)</h4>
              <ul>
                <li>Uses image segmentation to find connected regions of similar pixels</li>
                <li>Applies flood-fill algorithms to identify boundaries</li>
                <li>Groups pixels into regions based on color similarity and connectivity</li>
                <li>Works well for simple line art and coloring book images</li>
              </ul>
            </div>
            <div className="algorithm-box">
              <h4>ML Segmentation (Optional)</h4>
              <ul>
                <li>Uses SLIC (Simple Linear Iterative Clustering) superpixel segmentation</li>
                <li>Groups pixels into perceptually meaningful regions</li>
                <li>Better handles complex images with gradients and textures</li>
                <li>Can merge similar regions to reduce complexity</li>
              </ul>
            </div>
          </div>
        </section>

        <section className="educational-section">
          <h3>Step 3: Adjacency Detection</h3>
          <div className="content-block">
            <p>
              Once regions are identified, the system determines which regions are adjacent to each other 
              (share a boundary). This is crucial for building the graph:
            </p>
            <ul>
              <li><strong>Boundary Analysis:</strong> Examines the edges of each region to find neighboring regions</li>
              <li><strong>Contour Detection:</strong> Uses computer vision techniques to trace region boundaries</li>
              <li><strong>Multi-scale Detection:</strong> Handles thin lines and overlapping regions</li>
              <li><strong>Graph Construction:</strong> Creates a graph where each region is a node, and edges connect adjacent regions</li>
            </ul>
            <p className="note">
              <strong>Note:</strong> The resulting graph must be planar (can be drawn on a plane without edges crossing) 
              for the Four Color Theorem to guarantee a 4-color solution.
            </p>
          </div>
        </section>

        <section className="educational-section">
          <h3>Step 4: Graph Coloring Algorithms</h3>
          <div className="content-block">
            <p>
              The application uses several graph coloring algorithms, selecting the best one based on graph size and complexity:
            </p>
            
            <div className="algorithm-box">
              <h4>Welsh-Powell Algorithm</h4>
              <p>A greedy algorithm that works well for most cases:</p>
              <ol>
                <li>Sort nodes by degree (number of connections) in descending order</li>
                <li>For each node, assign the first available color that isn't used by any neighbor</li>
                <li>Continue until all nodes are colored</li>
              </ol>
              <p><strong>Time Complexity:</strong> O(V²) where V is the number of vertices</p>
            </div>

            <div className="algorithm-box">
              <h4>DSATUR (Degree of Saturation)</h4>
              <p>An improved greedy algorithm used for larger graphs:</p>
              <ol>
                <li>At each step, select the node with the highest saturation degree (number of different colors used by neighbors)</li>
                <li>If there's a tie, choose the node with the highest degree</li>
                <li>Assign the lowest-numbered color not used by neighbors</li>
              </ol>
              <p><strong>Time Complexity:</strong> O(V²) but often uses fewer colors than Welsh-Powell</p>
            </div>

            <div className="algorithm-box">
              <h4>Backtracking Algorithm</h4>
              <p>A complete search algorithm that guarantees optimal coloring for smaller graphs:</p>
              <ol>
                <li>Systematically tries all possible color assignments</li>
                <li>If a conflict is found, backtracks and tries a different color</li>
                <li>Continues until a valid coloring is found or all possibilities are exhausted</li>
              </ol>
              <p><strong>Time Complexity:</strong> O(4^V) in worst case - exponential, so only used for small graphs (&lt;100 nodes)</p>
            </div>
          </div>
        </section>

        <section className="educational-section">
          <h3>Step 5: Color Application</h3>
          <div className="content-block">
            <p>
              Once the graph is colored, the colors are applied back to the original image:
            </p>
            <ul>
              <li>Each region receives its assigned color from the palette</li>
              <li>Colors can be selected from predefined styles (vibrant, pastel, nature, etc.)</li>
              <li>Optional AI-powered color suggestions can recommend palettes based on image content</li>
              <li>Custom color palettes can be specified by the user</li>
            </ul>
            <p>
              The application supports both 4-color and optional 5-color modes. While the Four Color Theorem 
              guarantees that 4 colors are always sufficient for planar graphs, some non-planar graphs (which 
              can occur in complex images) may benefit from a 5th color.
            </p>
          </div>
        </section>

        <section className="educational-section">
          <h2>Why This Matters</h2>
          <div className="content-block">
            <p>
              The Four Color Theorem has practical applications beyond map coloring:
            </p>
            <ul>
              <li><strong>Automatic Coloring:</strong> As demonstrated in this application, it enables automatic coloring of line art and coloring book images</li>
              <li><strong>Resource Allocation:</strong> Used in scheduling problems where conflicting tasks need different resources</li>
              <li><strong>Register Allocation:</strong> In compiler design, assigning CPU registers to variables</li>
              <li><strong>Frequency Assignment:</strong> Assigning radio frequencies to avoid interference</li>
              <li><strong>Sudoku Solving:</strong> Graph coloring techniques can be applied to solve Sudoku puzzles</li>
            </ul>
          </div>
        </section>

        <section className="educational-section">
          <h2>Technical Details</h2>
          <div className="content-block">
            <h4>Graph Theory Concepts</h4>
            <dl>
              <dt><strong>Planar Graph:</strong></dt>
              <dd>A graph that can be drawn on a plane without any edges crossing. Maps naturally form planar graphs.</dd>
              
              <dt><strong>Chromatic Number:</strong></dt>
              <dd>The minimum number of colors needed to color a graph. For planar graphs, this is at most 4.</dd>
              
              <dt><strong>Adjacency:</strong></dt>
              <dd>Two vertices (regions) are adjacent if they share an edge (boundary) in the graph.</dd>
              
              <dt><strong>Graph Coloring:</strong></dt>
              <dd>The assignment of colors to vertices such that no two adjacent vertices have the same color.</dd>
            </dl>
          </div>
        </section>

        <section className="educational-section">
          <h2>Further Reading</h2>
          <div className="content-block">
            <p>If you're interested in learning more about the Four Color Theorem:</p>
            <ul>
              <li><a href="https://en.wikipedia.org/wiki/Four_color_theorem" target="_blank" rel="noopener noreferrer">Wikipedia: Four Color Theorem</a></li>
              <li><a href="https://mathworld.wolfram.com/Four-ColorTheorem.html" target="_blank" rel="noopener noreferrer">Wolfram MathWorld: Four-Color Theorem</a></li>
              <li>Graph Theory textbooks by authors like Bondy & Murty or Diestel</li>
              <li>Computer Science algorithms textbooks covering graph coloring</li>
            </ul>
          </div>
        </section>

        <footer className="educational-footer">
          <button className="back-button" onClick={onBack}>
            ← Back to App
          </button>
        </footer>
      </div>
    </div>
  );
}

export default EducationalPage;

