package device

import (
	"context"
	"sync"
)

// Task represents a task to be executed by the worker pool
type Task func()

// WorkerPool manages a pool of workers for concurrent task execution
type WorkerPool struct {
	size       int
	taskQueue  chan Task
	workers    []*Worker
	wg         sync.WaitGroup
	mu         sync.RWMutex
	closed     bool
	queueSize  int
}

// Worker represents a single worker
type Worker struct {
	id       int
	pool     *WorkerPool
	taskChan chan Task
	quit     chan bool
	active   bool
	mu       sync.RWMutex
}

// NewWorkerPool creates a new worker pool
func NewWorkerPool(size int) *WorkerPool {
	return &WorkerPool{
		size:      size,
		taskQueue: make(chan Task, size*2), // Buffer for queued tasks
		workers:   make([]*Worker, size),
		queueSize: 0,
	}
}

// Start starts the worker pool
func (wp *WorkerPool) Start(ctx context.Context) {
	wp.mu.Lock()
	defer wp.mu.Unlock()

	if wp.closed {
		return
	}

	// Create and start workers
	for i := 0; i < wp.size; i++ {
		worker := &Worker{
			id:       i,
			pool:     wp,
			taskChan: make(chan Task),
			quit:     make(chan bool),
			active:   false,
		}
		wp.workers[i] = worker
		wp.wg.Add(1)
		go worker.start(ctx)
	}

	// Start dispatcher
	go wp.dispatch(ctx)
}

// Submit submits a task to the worker pool
func (wp *WorkerPool) Submit(task Task) {
	wp.mu.RLock()
	defer wp.mu.RUnlock()

	if wp.closed {
		return
	}

	select {
	case wp.taskQueue <- task:
		wp.queueSize++
	default:
		// Queue is full, drop the task
		// In a production system, you might want to handle this differently
	}
}

// Size returns the number of workers in the pool
func (wp *WorkerPool) Size() int {
	wp.mu.RLock()
	defer wp.mu.RUnlock()
	return wp.size
}

// QueueSize returns the current queue size
func (wp *WorkerPool) QueueSize() int {
	wp.mu.RLock()
	defer wp.mu.RUnlock()
	return wp.queueSize
}

// ActiveWorkers returns the number of currently active workers
func (wp *WorkerPool) ActiveWorkers() int {
	wp.mu.RLock()
	defer wp.mu.RUnlock()

	active := 0
	for _, worker := range wp.workers {
		if worker.IsActive() {
			active++
		}
	}
	return active
}

// dispatch dispatches tasks to available workers
func (wp *WorkerPool) dispatch(ctx context.Context) {
	for {
		select {
		case task := <-wp.taskQueue:
			wp.mu.Lock()
			wp.queueSize--
			wp.mu.Unlock()

			// Find an available worker
			wp.assignTask(task)

		case <-ctx.Done():
			return
		}
	}
}

// assignTask assigns a task to an available worker
func (wp *WorkerPool) assignTask(task Task) {
	wp.mu.RLock()
	defer wp.mu.RUnlock()

	// Round-robin assignment
	for _, worker := range wp.workers {
		select {
		case worker.taskChan <- task:
			return
		default:
			// Worker is busy, try next one
			continue
		}
	}

	// If all workers are busy, this will block until one becomes available
	// In a production system, you might want to handle this differently
	go func() {
		wp.workers[0].taskChan <- task
	}()
}

// Close closes the worker pool
func (wp *WorkerPool) Close() {
	wp.mu.Lock()
	defer wp.mu.Unlock()

	if wp.closed {
		return
	}

	wp.closed = true
	close(wp.taskQueue)

	// Stop all workers
	for _, worker := range wp.workers {
		worker.stop()
	}

	// Wait for all workers to finish
	wp.wg.Wait()
}

// start starts the worker
func (w *Worker) start(ctx context.Context) {
	defer w.pool.wg.Done()

	for {
		select {
		case task := <-w.taskChan:
			w.setActive(true)
			task()
			w.setActive(false)

		case <-w.quit:
			return

		case <-ctx.Done():
			return
		}
	}
}

// stop stops the worker
func (w *Worker) stop() {
	close(w.quit)
}

// IsActive returns whether the worker is currently active
func (w *Worker) IsActive() bool {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.active
}

// setActive sets the worker's active status
func (w *Worker) setActive(active bool) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.active = active
}